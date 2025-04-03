#!/usr/bin/env python3
"""
Minimal training script that correctly saves TensorFlow models with .keras extension
"""

import os
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_industry_model(industry, data_dir="industry_data_from_db"):
    """Train a model for the given industry with proper file extension."""
    try:
        # Construct paths for input and output
        csv_file = os.path.join(data_dir, f"{industry.lower().replace(' ', '_')}_training.csv")
        
        # Check if training data exists
        if not os.path.exists(csv_file):
            logger.error(f"Training data file not found: {csv_file}")
            return False
        
        # Load data
        data = pd.read_csv(csv_file)
        
        if data.empty:
            logger.error(f"No data available for {industry}")
            return False
            
        logger.info(f"Training model for {industry} with {len(data)} records")
        
        # Select features
        potential_features = [
            'revenue', 'operating_margin', 'net_margin', 'roa', 'roe',
            'historical_growth_mean', 'historical_growth', 'debt_to_equity'
        ]
        
        # Use features that exist in the data
        features = [f for f in potential_features if f in data.columns]
        
        if len(features) < 2:
            logger.error(f"Not enough features available for {industry}")
            return False
            
        logger.info(f"Using features: {features}")
        
        # Prepare features and target
        X = data[features].copy()
        
        # Ensure future_6m_return exists
        if 'future_6m_return' not in data.columns:
            logger.warning(f"No future_6m_return in data, using synthetic values")
            # Generate synthetic target based on features
            data['future_6m_return'] = 0.05 + 0.1 * data['operating_margin'] - 0.05 * np.random.random(len(data))
        
        y = data['future_6m_return']
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(0.05)  # Default expected return
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create simple model
        input_shape = (X.shape[1],)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),  # Proper input layer
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        
        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        history = model.fit(
            X_scaled, y,
            epochs=50,
            batch_size=min(16, len(X)),
            validation_split=0.2 if len(X) > 10 else 0,
            verbose=1
        )
        
        # Create models directory if it doesn't exist
        model_dir = os.path.join(data_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model WITH .keras extension
        model_path = os.path.join(model_dir, f"{industry.lower().replace(' ', '_')}_model.keras")
        scaler_path = os.path.join(model_dir, f"{industry.lower().replace(' ', '_')}_scaler.pkl")
        
        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
            
        logger.info(f"Model saved to {model_path}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training Loss for {industry}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(data_dir, f"{industry.lower().replace(' ', '_')}_loss.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Successfully trained model for {industry}")
        return True
        
    except Exception as e:
        logger.error(f"Error training model for {industry}: {str(e)}", exc_info=True)
        return False

def main():
    """Train models for all available industries."""
    data_dir = "industry_data_from_db"
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found")
        return False
    
    # Find training CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_training.csv')]
    
    if not csv_files:
        logger.error(f"No training CSV files found in {data_dir}")
        return False
    
    logger.info(f"Found {len(csv_files)} training data files")
    
    # Process each industry
    successful = []
    for csv_file in csv_files:
        industry = csv_file.replace('_training.csv', '').replace('_', ' ')
        logger.info(f"Processing {industry}")
        
        if train_industry_model(industry, data_dir):
            successful.append(industry)
    
    # Report success
    logger.info(f"Trained models for {len(successful)}/{len(csv_files)} industries")
    logger.info(f"Successful industries: {', '.join(successful)}")
    
    return True

if __name__ == "__main__":
    main()
