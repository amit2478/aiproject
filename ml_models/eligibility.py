import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EligibilityAssessor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.model_path = Path("models/eligibility_model.joblib")
        self.scaler_path = Path("models/scaler.joblib")
        self.imputer_path = Path("models/imputer.joblib")
        
        # Create models directory if it doesn't exist
        Path("models").mkdir(exist_ok=True)
        
        # Initialize or load model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the trained model."""
        try:
            if self.model_path.exists():
                logger.info("Loading existing model...")
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.imputer = joblib.load(self.imputer_path)
            else:
                logger.info("Initializing new model...")
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                # Train with synthetic data
                self._train_with_synthetic_data()
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _train_with_synthetic_data(self):
        """Train the model with synthetic data."""
        try:
            # Generate synthetic training data
            n_samples = 1000
            X = np.random.rand(n_samples, 4)  # 4 features
            y = np.random.randint(0, 2, n_samples)  # Binary classification
            
            # Preprocess data
            X = self.imputer.fit_transform(X)
            X = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X, y)
            
            # Save model and preprocessors
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.imputer, self.imputer_path)
            
            logger.info("Model trained and saved successfully")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def preprocess_features(self, features):
        """Preprocess input features."""
        try:
            # Convert features to numpy array
            X = np.array(features).reshape(1, -1)
            
            # Apply preprocessing
            X = self.imputer.transform(X)
            X = self.scaler.transform(X)
            
            return X
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            raise
    
    def assess_eligibility(self, application_data):
        """Assess eligibility based on application data."""
        try:
            # Extract features
            features = [
                application_data['income'],
                application_data['expenses'],
                application_data['dependents'],
                1 if application_data['employment_status'] == 'Employed' else 0
            ]
            
            # Preprocess features
            X = self.preprocess_features(features)
            
            # Get prediction probability
            proba = self.model.predict_proba(X)[0]
            eligibility_score = proba[1] * 100  # Convert to percentage
            
            # Determine risk level
            risk_level = "Low" if eligibility_score >= 80 else "Medium" if eligibility_score >= 50 else "High"
            
            # Calculate recommended support
            base_support = 1000  # Base support amount
            recommended_support = base_support * (eligibility_score / 100)
            
            # Determine duration
            duration_months = 6 if eligibility_score >= 80 else 3 if eligibility_score >= 50 else 1
            
            # Calculate factor scores
            factors = [
                {
                    "name": "Income Level",
                    "score": min(100, max(0, (application_data['income'] / 5000) * 100)),
                    "impact": "Positive" if application_data['income'] < 3000 else "Neutral"
                },
                {
                    "name": "Employment Status",
                    "score": 90 if application_data['employment_status'] == 'Employed' else 50,
                    "impact": "Positive" if application_data['employment_status'] == 'Employed' else "Neutral"
                },
                {
                    "name": "Dependents",
                    "score": min(100, max(0, application_data['dependents'] * 20)),
                    "impact": "Positive" if application_data['dependents'] > 0 else "Neutral"
                },
                {
                    "name": "Expenses",
                    "score": min(100, max(0, (1 - application_data['expenses'] / application_data['income']) * 100)),
                    "impact": "Positive" if application_data['expenses'] < application_data['income'] * 0.7 else "Neutral"
                }
            ]
            
            return {
                "eligibility_score": round(eligibility_score, 2),
                "risk_level": risk_level,
                "recommended_support": round(recommended_support, 2),
                "duration_months": duration_months,
                "factors": factors
            }
        except Exception as e:
            logger.error(f"Error assessing eligibility: {str(e)}")
            raise

# Create singleton instance
eligibility_assessor = EligibilityAssessor() 