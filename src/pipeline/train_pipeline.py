import os
import sys
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.common import read_yaml, create_directories
from src.utils.data_ops import load_data, preprocess_data
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
import joblib

"""
Training Pipeline for MLOps Project
"""

import mlflow.sklearn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainPipeline:
    """Main training pipeline class"""
    
    def __init__(self, config_path: str):
        """Initialize the training pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = read_yaml(config_path)
        self.artifacts_dir = Path(self.config['artifacts']['root_dir'])
        self.model_dir = self.artifacts_dir / 'models'
        self.report_dir = self.artifacts_dir / 'reports'
        
        # Create directories
        create_directories([self.model_dir, self.report_dir])
        
        # Initialize MLflow
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
    def load_data(self):
        """Load training data"""
        logger.info("Loading data...")
        data_path = self.config['data']['train_data_path']
        df = load_data(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def preprocess(self, df: pd.DataFrame):
        """Preprocess the data"""
        logger.info("Preprocessing data...")
        target_column = self.config['data']['target_column']
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Training set size: {X_train_scaled.shape[0]}")
        logger.info(f"Test set size: {X_test_scaled.shape[0]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def train(self, X_train, y_train):
        """Train the model"""
        logger.info("Training model...")
        
        model_params = self.config['model']['params']
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(model_params)
            
            # Train model
            model = train_model(X_train, y_train, model_params)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=self.config['mlflow']['model_name']
            )
            
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Model trained. MLflow Run ID: {run_id}")
            
        return model
    
    def evaluate(self, model, X_test, y_test):
        """Evaluate the model"""
        logger.info("Evaluating model...")
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save_model(self, model, scaler):
        """Save the trained model and artifacts"""
        logger.info("Saving model and artifacts...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.model_dir / f"model_{timestamp}.pkl"
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = self.model_dir / f"scaler_{timestamp}.pkl"
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Scaler saved to: {scaler_path}")
        
        return str(model_path), str(scaler_path)
    
    def run(self):
        """Execute the full training pipeline"""
        logger.info("=" * 50)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 50)
        
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess data
            X_train, X_test, y_train, y_test, scaler = self.preprocess(df)
            
            # Train model
            model = self.train(X_train, y_train)
            
            # Evaluate model
            metrics = self.evaluate(model, X_test, y_test)
            
            # Save model
            model_path, scaler_path = self.save_model(model, scaler)
            
            logger.info("=" * 50)
            logger.info("Training Pipeline Completed Successfully!")
            logger.info("=" * 50)
            
            return {
                'status': 'success',
                'model_path': model_path,
                'scaler_path': scaler_path,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Training Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = TrainPipeline(args.config)
    results = pipeline.run()
    
    print(f"\nTraining Results: {results}")


if __name__ == '__main__':
    main()