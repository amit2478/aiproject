from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path
from sqlalchemy.orm import Session
from database.models import SessionLocal, Application, Document, Assessment, ChatHistory
from ml_models.eligibility import eligibility_assessor
from data_ingestion.document_processor import document_processor
from agents.agent_system import agent_system

# Initialize FastAPI app
app = FastAPI(
    title="Government Social Support AI API",
    description="API for social support eligibility assessment and document processing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class ApplicationData(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: str
    income: float
    employment_status: str
    expenses: float
    dependents: int

class AssessmentResult(BaseModel):
    eligibility_score: float
    risk_level: str
    recommended_support: float
    duration_months: int
    factors: List[dict]

@app.post("/api/application/submit")
async def submit_application(data: ApplicationData, db: Session = Depends(get_db)):
    """Submit a new application for social support."""
    try:
        # Create new application
        application = Application(
            application_id=f"APP{datetime.now().strftime('%Y%m%d%H%M%S')}",
            first_name=data.first_name,
            last_name=data.last_name,
            email=data.email,
            phone=data.phone,
            income=data.income,
            employment_status=data.employment_status,
            expenses=data.expenses,
            dependents=data.dependents,
            status="pending"
        )
        
        db.add(application)
        db.commit()
        db.refresh(application)
        
        return {"application_id": application.application_id, "status": "submitted"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload/{application_id}")
async def upload_documents(
    application_id: str,
    bank_statement: Optional[UploadFile] = File(None),
    id_proof: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Upload supporting documents for an application."""
    try:
        # Get application
        application = db.query(Application).filter(Application.application_id == application_id).first()
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        
        # Process and save documents
        if bank_statement:
            content = await bank_statement.read()
            result = document_processor.process_document(content, bank_statement.filename, "bank_statement")
            
            document = Document(
                application_id=application.id,
                document_type="bank_statement",
                file_path=str(Path("uploads") / f"{application_id}_bank_statement.pdf"),
                processed=True,
                extracted_data=result.get("extracted_data", {})
            )
            db.add(document)
        
        if id_proof:
            content = await id_proof.read()
            result = document_processor.process_document(content, id_proof.filename, "id_proof")
            
            document = Document(
                application_id=application.id,
                document_type="id_proof",
                file_path=str(Path("uploads") / f"{application_id}_id_proof.pdf"),
                processed=True,
                extracted_data=result.get("extracted_data", {})
            )
            db.add(document)
        
        db.commit()
        return {"status": "success", "message": "Documents uploaded and processed successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/assessment/{application_id}")
async def get_assessment(application_id: str, db: Session = Depends(get_db)):
    """Get eligibility assessment results for an application."""
    try:
        # Get application
        application = db.query(Application).filter(Application.application_id == application_id).first()
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        
        # Get or create assessment
        assessment = db.query(Assessment).filter(Assessment.application_id == application.id).first()
        if not assessment:
            # Process application with agent system
            result = agent_system.process_application({
                "income": application.income,
                "expenses": application.expenses,
                "dependents": application.dependents,
                "employment_status": application.employment_status
            })
            
            # Create assessment
            assessment = Assessment(
                application_id=application.id,
                eligibility_score=result["eligibility_score"],
                risk_level=result["risk_level"],
                recommended_support=result["recommended_support"],
                duration_months=result["duration_months"],
                factors=result["factors"]
            )
            db.add(assessment)
            db.commit()
            db.refresh(assessment)
        
        return {
            "eligibility_score": assessment.eligibility_score,
            "risk_level": assessment.risk_level,
            "recommended_support": assessment.recommended_support,
            "duration_months": assessment.duration_months,
            "factors": assessment.factors
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/applications")
async def list_applications(
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all applications with optional filtering."""
    try:
        query = db.query(Application)
        
        if status:
            query = query.filter(Application.status == status)
        if start_date:
            query = query.filter(Application.submission_date >= start_date)
        if end_date:
            query = query.filter(Application.submission_date <= end_date)
        
        applications = query.all()
        return [
            {
                "application_id": app.application_id,
                "name": f"{app.first_name} {app.last_name}",
                "status": app.status,
                "submission_date": app.submission_date.isoformat(),
                "assessment": {
                    "eligibility_score": app.assessment.eligibility_score if app.assessment else None,
                    "risk_level": app.assessment.risk_level if app.assessment else None
                } if app.assessment else None
            }
            for app in applications
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_assistant(message: str, application_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Chat with the AI assistant."""
    try:
        # Get application context if provided
        context = {}
        if application_id:
            application = db.query(Application).filter(Application.application_id == application_id).first()
            if application:
                context = {
                    "application_id": application.application_id,
                    "name": f"{application.first_name} {application.last_name}",
                    "status": application.status
                }
        
        # Process with agent system
        response = agent_system.process_application({
            "message": message,
            "context": context
        })
        
        # Save chat history
        if application_id:
            chat = ChatHistory(
                application_id=application.id,
                message=message,
                response=response["message"],
                confidence=response.get("confidence", 0.0)
            )
            db.add(chat)
            db.commit()
        
        return response
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 