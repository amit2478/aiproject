import pytesseract
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import cv2
import numpy as np
from PIL import Image
import io
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    def process_pdf(self, file_content: bytes, filename: str) -> dict:
        """Process PDF document and extract text."""
        try:
            # Save PDF temporarily
            temp_path = self.upload_dir / filename
            with open(temp_path, "wb") as f:
                f.write(file_content)
            
            # Extract text from PDF
            text = extract_text(temp_path, laparams=LAParams())
            
            # Clean up
            os.remove(temp_path)
            
            return {
                "text": text,
                "type": "pdf",
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                "text": "",
                "type": "pdf",
                "status": "error",
                "error": str(e)
            }
    
    def process_image(self, file_content: bytes, filename: str) -> dict:
        """Process image document using OCR."""
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Perform OCR
            text = pytesseract.image_to_string(thresh)
            
            return {
                "text": text,
                "type": "image",
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                "text": "",
                "type": "image",
                "status": "error",
                "error": str(e)
            }
    
    def extract_bank_statement_data(self, text: str) -> dict:
        """Extract relevant information from bank statement text."""
        try:
            # Mock extraction - in real implementation, use regex or NLP
            data = {
                "account_balance": 0.0,
                "transactions": [],
                "statement_date": "",
                "account_number": ""
            }
            
            # Simple pattern matching
            lines = text.split("\n")
            for line in lines:
                if "balance" in line.lower():
                    try:
                        balance = float(line.split()[-1].replace(",", ""))
                        data["account_balance"] = balance
                    except:
                        pass
                elif "date" in line.lower():
                    data["statement_date"] = line.split()[-1]
                elif "account" in line.lower():
                    data["account_number"] = line.split()[-1]
            
            return data
        except Exception as e:
            logger.error(f"Error extracting bank statement data: {str(e)}")
            return {
                "account_balance": 0.0,
                "transactions": [],
                "statement_date": "",
                "account_number": "",
                "error": str(e)
            }
    
    def extract_id_data(self, text: str) -> dict:
        """Extract relevant information from ID document text."""
        try:
            # Mock extraction - in real implementation, use regex or NLP
            data = {
                "name": "",
                "id_number": "",
                "date_of_birth": "",
                "address": ""
            }
            
            # Simple pattern matching
            lines = text.split("\n")
            for line in lines:
                if "name" in line.lower():
                    data["name"] = line.split(":", 1)[-1].strip()
                elif "id" in line.lower():
                    data["id_number"] = line.split(":", 1)[-1].strip()
                elif "dob" in line.lower() or "birth" in line.lower():
                    data["date_of_birth"] = line.split(":", 1)[-1].strip()
                elif "address" in line.lower():
                    data["address"] = line.split(":", 1)[-1].strip()
            
            return data
        except Exception as e:
            logger.error(f"Error extracting ID data: {str(e)}")
            return {
                "name": "",
                "id_number": "",
                "date_of_birth": "",
                "address": "",
                "error": str(e)
            }
    
    def process_document(self, file_content: bytes, filename: str, doc_type: str) -> dict:
        """Process document based on type."""
        try:
            # Process based on file type
            if filename.lower().endswith('.pdf'):
                result = self.process_pdf(file_content, filename)
            else:
                result = self.process_image(file_content, filename)
            
            if result["status"] == "success":
                # Extract specific data based on document type
                if doc_type == "bank_statement":
                    extracted_data = self.extract_bank_statement_data(result["text"])
                elif doc_type == "id_proof":
                    extracted_data = self.extract_id_data(result["text"])
                else:
                    extracted_data = {"text": result["text"]}
                
                result["extracted_data"] = extracted_data
            
            return result
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

# Create singleton instance
document_processor = DocumentProcessor() 