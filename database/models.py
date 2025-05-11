from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create database engine with local PostgreSQL configuration
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/social_support_db"
engine = create_engine(DATABASE_URL)

# Create base class
Base = declarative_base()

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ... rest of the code remains the same ... 