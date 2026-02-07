"""
Database Module
===============
This module handles storage and retrieval of prediction results.
Stores predictions in SQLite database with timestamps and metadata.

Author: B.Tech Major Project
Date: 2026
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionDatabase:
    """
    Manages storage and retrieval of prediction results in SQLite database.
    
    Database Schema:
    - predictions: stores individual prediction results
    - metadata: stores dataset and model information
    """
    
    def __init__(self, db_path='./database/predictions.db'):
        """
        Initialize the PredictionDatabase.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        
        # Create directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"Initialized PredictionDatabase at {db_path}")
    
    def _initialize_database(self):
        """
        Initialize SQLite database with required tables.
        """
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            
            # Create predictions table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_path TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    probabilities TEXT NOT NULL,
                    actual_class TEXT,
                    correct BOOLEAN,
                    notes TEXT
                )
            ''')
            
            # Create metadata table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.connection.commit()
            logger.info("Database tables created successfully")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def store_prediction(self, image_path, predicted_class, confidence,
                        probabilities, actual_class=None, notes=None):
        """
        Store a single prediction result.
        
        Args:
            image_path (str): Path to the image
            predicted_class (str): Predicted class name
            confidence (float): Confidence score
            probabilities (dict): Probability for each class
            actual_class (str): Ground truth class (optional)
            notes (str): Additional notes
        
        Returns:
            int: ID of inserted record
        """
        try:
            # Check if prediction is correct
            correct = None
            if actual_class:
                correct = predicted_class == actual_class
            
            # Convert probabilities dict to JSON string
            probabilities_json = json.dumps(probabilities)
            
            # Insert record
            self.cursor.execute('''
                INSERT INTO predictions
                (image_path, predicted_class, confidence, probabilities, 
                 actual_class, correct, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (image_path, predicted_class, confidence, probabilities_json,
                  actual_class, correct, notes))
            
            self.connection.commit()
            
            record_id = self.cursor.lastrowid
            logger.info(f"Prediction stored with ID: {record_id}")
            
            return record_id
        
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            return None
    
    def store_batch_predictions(self, predictions):
        """
        Store multiple prediction results.
        
        Args:
            predictions (list): List of prediction dictionaries
        
        Returns:
            list: List of inserted record IDs
        """
        logger.info(f"Storing {len(predictions)} predictions...")
        
        record_ids = []
        for i, pred in enumerate(predictions):
            if (i + 1) % 50 == 0:
                logger.info(f"Stored {i + 1}/{len(predictions)} predictions")
            
            record_id = self.store_prediction(
                image_path=pred.get('image_path', ''),
                predicted_class=pred.get('class', ''),
                confidence=pred.get('confidence', 0.0),
                probabilities=pred.get('probabilities', {}),
                actual_class=pred.get('actual_class', None),
                notes=pred.get('notes', None)
            )
            
            if record_id:
                record_ids.append(record_id)
        
        logger.info(f"Batch storage completed. {len(record_ids)} records inserted.")
        return record_ids
    
    def get_prediction(self, prediction_id):
        """
        Get a single prediction by ID.
        
        Args:
            prediction_id (int): Prediction ID
        
        Returns:
            dict: Prediction record
        """
        try:
            self.cursor.execute(
                'SELECT * FROM predictions WHERE id = ?',
                (prediction_id,)
            )
            
            columns = [desc[0] for desc in self.cursor.description]
            row = self.cursor.fetchone()
            
            if row:
                record = dict(zip(columns, row))
                # Parse probabilities JSON
                record['probabilities'] = json.loads(record['probabilities'])
                return record
            else:
                logger.warning(f"Prediction with ID {prediction_id} not found")
                return None
        
        except Exception as e:
            logger.error(f"Error retrieving prediction: {str(e)}")
            return None
    
    def get_all_predictions(self, limit=None):
        """
        Get all predictions from database.
        
        Args:
            limit (int): Maximum number of records to retrieve
        
        Returns:
            list: List of prediction records
        """
        try:
            query = 'SELECT * FROM predictions'
            if limit:
                query += f' LIMIT {limit}'
            
            self.cursor.execute(query)
            
            columns = [desc[0] for desc in self.cursor.description]
            rows = self.cursor.fetchall()
            
            records = []
            for row in rows:
                record = dict(zip(columns, row))
                record['probabilities'] = json.loads(record['probabilities'])
                records.append(record)
            
            logger.info(f"Retrieved {len(records)} predictions")
            return records
        
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            return []
    
    def get_predictions_by_class(self, class_name):
        """
        Get all predictions for a specific class.
        
        Args:
            class_name (str): Class name
        
        Returns:
            list: List of prediction records
        """
        try:
            self.cursor.execute(
                'SELECT * FROM predictions WHERE predicted_class = ?',
                (class_name,)
            )
            
            columns = [desc[0] for desc in self.cursor.description]
            rows = self.cursor.fetchall()
            
            records = []
            for row in rows:
                record = dict(zip(columns, row))
                record['probabilities'] = json.loads(record['probabilities'])
                records.append(record)
            
            logger.info(f"Retrieved {len(records)} predictions for class {class_name}")
            return records
        
        except Exception as e:
            logger.error(f"Error retrieving predictions by class: {str(e)}")
            return []
    
    def get_statistics(self):
        """
        Get statistics from predictions database.
        
        Returns:
            dict: Statistical summary
        """
        try:
            stats = {}
            
            # Total predictions
            self.cursor.execute('SELECT COUNT(*) FROM predictions')
            stats['total_predictions'] = self.cursor.fetchone()[0]
            
            # Count by class
            self.cursor.execute('''
                SELECT predicted_class, COUNT(*) as count
                FROM predictions
                GROUP BY predicted_class
            ''')
            stats['class_counts'] = dict(self.cursor.fetchall())
            
            # Average confidence
            self.cursor.execute('''
                SELECT predicted_class, AVG(confidence) as avg_confidence
                FROM predictions
                GROUP BY predicted_class
            ''')
            stats['avg_confidence'] = dict(self.cursor.fetchall())
            
            # Overall accuracy (if actual_class is available)
            self.cursor.execute('''
                SELECT COUNT(*) as total, SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct
                FROM predictions
                WHERE actual_class IS NOT NULL
            ''')
            result = self.cursor.fetchone()
            if result[0] > 0:
                stats['accuracy'] = result[1] / result[0] if result[1] else 0
            
            logger.info("Statistics computed")
            return stats
        
        except Exception as e:
            logger.error(f"Error computing statistics: {str(e)}")
            return {}
    
    def export_to_csv(self, output_path='./predictions.csv'):
        """
        Export all predictions to CSV file.
        
        Args:
            output_path (str): Path to save CSV file
        
        Returns:
            bool: Success status
        """
        try:
            predictions = self.get_all_predictions()
            
            if not predictions:
                logger.warning("No predictions to export")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(predictions)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            logger.info(f"Predictions exported to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return False
    
    def export_to_excel(self, output_path='./predictions.xlsx'):
        """
        Export all predictions to Excel file.
        
        Args:
            output_path (str): Path to save Excel file
        
        Returns:
            bool: Success status
        """
        try:
            predictions = self.get_all_predictions()
            
            if not predictions:
                logger.warning("No predictions to export")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(predictions)
            
            # Save to Excel
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Predictions')
            
            logger.info(f"Predictions exported to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}")
            return False
    
    def store_metadata(self, key, value):
        """
        Store metadata (model info, dataset info, etc.).
        
        Args:
            key (str): Metadata key
            value (str): Metadata value
        """
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES (?, ?)
            ''', (key, value))
            
            self.connection.commit()
            logger.info(f"Metadata stored: {key} = {value}")
        
        except Exception as e:
            logger.error(f"Error storing metadata: {str(e)}")
    
    def get_metadata(self, key):
        """
        Retrieve metadata value.
        
        Args:
            key (str): Metadata key
        
        Returns:
            str: Metadata value
        """
        try:
            self.cursor.execute(
                'SELECT value FROM metadata WHERE key = ?',
                (key,)
            )
            result = self.cursor.fetchone()
            return result[0] if result else None
        
        except Exception as e:
            logger.error(f"Error retrieving metadata: {str(e)}")
            return None
    
    def clear_predictions(self):
        """
        Clear all predictions from database.
        
        WARNING: This action cannot be undone!
        
        Returns:
            bool: Success status
        """
        try:
            self.cursor.execute('DELETE FROM predictions')
            self.connection.commit()
            logger.warning("All predictions cleared from database")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing predictions: {str(e)}")
            return False
    
    def close(self):
        """
        Close database connection.
        """
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __del__(self):
        """
        Destructor to ensure database is closed.
        """
        self.close()


def main():
    """
    Example usage of PredictionDatabase.
    """
    print("Database Module Test")
    print("=" * 60)
    
    # Initialize database
    print("Initializing database...")
    db = PredictionDatabase('./test_predictions.db')
    
    # Store sample predictions
    print("\nStoring sample predictions...")
    sample_predictions = [
        {
            'image_path': '/path/to/image1.png',
            'class': 'UAV',
            'confidence': 0.95,
            'probabilities': {'UAV': 0.95, 'Bird': 0.05},
            'actual_class': 'UAV'
        },
        {
            'image_path': '/path/to/image2.png',
            'class': 'Bird',
            'confidence': 0.87,
            'probabilities': {'UAV': 0.13, 'Bird': 0.87},
            'actual_class': 'Bird'
        },
        {
            'image_path': '/path/to/image3.png',
            'class': 'UAV',
            'confidence': 0.72,
            'probabilities': {'UAV': 0.72, 'Bird': 0.28},
            'actual_class': 'Bird'
        }
    ]
    
    db.store_batch_predictions(sample_predictions)
    
    # Retrieve and display statistics
    print("\nDatabase Statistics:")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export to CSV and Excel
    print("\nExporting predictions...")
    db.export_to_csv('./test_predictions.csv')
    db.export_to_excel('./test_predictions.xlsx')
    
    print("\nDatabase test completed!")
    
    db.close()


if __name__ == "__main__":
    main()
