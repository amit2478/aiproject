import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from dotenv import load_dotenv
from agents.agent_system import agent_system

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Government Social Support AI System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home Dashboard", "New Application", "Eligibility Assessment", "Chat Assistant", "Admin Panel"]
)

# Mock data generation
def get_mock_data():
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    return {
        'dates': dates,
        'applications': np.random.randint(10, 50, 30),
        'approvals': np.random.randint(5, 30, 30),
        'rejections': np.random.randint(2, 15, 30),
        'pending': np.random.randint(3, 20, 30)
    }

# Home Dashboard
if page == "Home Dashboard":
    st.title("🏛️ Government Social Support AI System")
    st.subheader("Home Dashboard")
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Applications", "1,234", "+12%")
    with col2:
        st.metric("Pending Reviews", "45", "-5%")
    with col3:
        st.metric("Approval Rate", "78%", "+3%")
    
    # Application trends
    st.subheader("Application Trends")
    data = get_mock_data()
    fig = px.line(data, x='dates', y=['applications', 'approvals'],
                  title='Daily Applications and Approvals')
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent applications
    st.subheader("Recent Applications")
    recent_apps = pd.DataFrame({
        'ID': ['APP001', 'APP002', 'APP003'],
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'Status': ['Pending', 'Approved', 'Under Review'],
        'Date': ['2024-03-10', '2024-03-09', '2024-03-08']
    })
    st.dataframe(recent_apps)

# New Application
elif page == "New Application":
    st.title("New Application Submission")
    
    with st.form("application_form"):
        # Personal Information
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name")
            email = st.text_input("Email")
        with col2:
            last_name = st.text_input("Last Name")
            phone = st.text_input("Phone Number")
        
        # Financial Information
        st.subheader("Financial Information")
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("Monthly Income", min_value=0)
            employment = st.selectbox("Employment Status", 
                                   ["Employed", "Self-employed", "Unemployed"])
        with col2:
            expenses = st.number_input("Monthly Expenses", min_value=0)
            dependents = st.number_input("Number of Dependents", min_value=0)
        
        # Document Upload
        st.subheader("Required Documents")
        bank_statement = st.file_uploader("Bank Statement (PDF)", type=['pdf'])
        id_proof = st.file_uploader("ID Proof (PDF/Image)", type=['pdf', 'png', 'jpg'])
        
        # Submit button
        submitted = st.form_submit_button("Submit Application")
        if submitted:
            st.success("Application submitted successfully!")

# Eligibility Assessment
elif page == "Eligibility Assessment":
    st.title("Eligibility Assessment Results")
    
    # Search application
    app_id = st.text_input("Enter Application ID")
    if app_id:
        # Mock assessment results
        st.subheader("Assessment Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Eligibility Score", "85%")
            st.metric("Risk Level", "Low")
        with col2:
            st.metric("Recommended Support", "$1,500/month")
            st.metric("Duration", "6 months")
        
        # Detailed breakdown
        st.subheader("Assessment Breakdown")
        factors = pd.DataFrame({
            'Factor': ['Income Level', 'Employment Status', 'Dependents', 'Expenses'],
            'Score': [85, 90, 75, 80],
            'Impact': ['Positive', 'Positive', 'Neutral', 'Positive']
        })
        st.dataframe(factors)
        
        # Visualization
        fig = px.bar(factors, x='Factor', y='Score',
                    title='Eligibility Factors')
        st.plotly_chart(fig, use_container_width=True)

# Chat Assistant
elif page == "Chat Assistant":
    st.title("AI Chat Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("How can I help you?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from agent system
        try:
            # Process the message with our agent system
            response = agent_system.process_application({
                "message": prompt,
                "context": {
                    "chat_history": st.session_state.messages[-5:] if len(st.session_state.messages) > 5 else st.session_state.messages
                }
            })
            
            # Extract the response message
            if isinstance(response, dict):
                response_text = response.get("message", "I apologize, but I couldn't process your request at the moment.")
            else:
                response_text = str(response)
            
            # Display AI response
            with st.chat_message("assistant"):
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
        except Exception as e:
            error_message = f"I apologize, but I encountered an error: {str(e)}"
            with st.chat_message("assistant"):
                st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Admin Panel
elif page == "Admin Panel":
    st.title("Admin Panel")
    
    # Authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        with st.form("admin_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                # Mock authentication
                if username == "admin" and password == "admin":
                    st.session_state.authenticated = True
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials")
    else:
        # Admin dashboard
        st.subheader("Application Review Queue")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox("Status", ["All", "Pending", "Approved", "Rejected"])
        with col2:
            date_filter = st.date_input("Date Range", [datetime.now()])
        
        # Applications table
        applications = pd.DataFrame({
            'ID': ['APP001', 'APP002', 'APP003', 'APP004'],
            'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
            'Status': ['Pending', 'Approved', 'Rejected', 'Pending'],
            'Date': ['2024-03-10', '2024-03-09', '2024-03-08', '2024-03-07'],
            'Score': [85, 92, 45, 78]
        })
        st.dataframe(applications)
        
        # Analytics
        st.subheader("Analytics")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(applications, names='Status', title='Application Status Distribution')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(applications, x='Score', title='Eligibility Score Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Logout button
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.experimental_rerun()

if __name__ == "__main__":
    pass 