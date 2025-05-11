# Government Social Support AI System

An intelligent system for automating financial and economic enablement support assessment using AI and machine learning.

## Features

- Interactive Streamlit dashboard for application management
- Multimodal data ingestion (structured and unstructured)
- AI-powered eligibility assessment
- GenAI Chat Assistant for applicant queries
- Admin panel for reviewing recommendations
- Local LLM integration with Ollama
- Vector storage with ChromaDB

## Architecture

The system consists of the following main components:

1. **Frontend (Streamlit)**
   - Home Dashboard
   - Application Submission
   - Eligibility Assessment
   - Chat Interface
   - Admin Panel

2. **Backend Services**
   - FastAPI for model serving
   - PostgreSQL for data storage
   - ChromaDB for vector storage
   - Ollama for local LLM hosting

3. **AI Components**
   - RandomForest Classifier for eligibility assessment
   - LangGraph for agent orchestration
   - Nomic embeddings for document processing
   - OCR processing with Tesseract

## Setup Instructions

1. **Prerequisites**
   - Python 3.9+
   - PostgreSQL
   - Tesseract OCR
   - Ollama (for local LLM)

2. **Installation**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd government-social-support-ai

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Set up environment variables
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Database Setup**
   ```bash
   # Create PostgreSQL database
   createdb social_support_db
   
   # Run migrations
   python scripts/setup_db.py
   ```

4. **Start Services**
   ```bash
   # Start Ollama (in a separate terminal)
   ollama serve

   # Start FastAPI backend
   uvicorn api.main:app --reload

   # Start Streamlit frontend
   streamlit run app.py
   ```

## Project Structure

```
├── app.py                 # Main Streamlit application
├── api/                   # FastAPI backend
├── data_ingestion/        # Data processing modules
├── ml_models/            # ML models and pipelines
├── agents/               # LangGraph agents
├── database/             # Database models and migrations
├── ui/                   # Streamlit UI components
├── utils/                # Utility functions
└── tests/                # Test suite
```

## Development

- Follow PEP 8 style guide
- Write unit tests for new features
- Document all functions and classes
- Use type hints

## Security Considerations

- All data is stored locally
- No external API calls for sensitive data
- Role-based access control
- Data encryption at rest

## Future Improvements

1. **Scalability**
   - Implement caching layer
   - Add load balancing
   - Optimize database queries

2. **Features**
   - Real-time notifications
   - Mobile app integration
   - Advanced analytics dashboard
   - Multi-language support

3. **AI Enhancements**
   - Fine-tune models on domain data
   - Add more specialized agents
   - Implement A/B testing framework

## License

This project is licensed under the MIT License - see the LICENSE file for details. 