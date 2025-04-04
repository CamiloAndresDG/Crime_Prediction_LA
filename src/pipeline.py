from data.data_ingestion import CrimeDataIngestion
from processing.data_processor import CrimeDataProcessor
from models.crime_predictor import CrimePredictionModel
import logging
from datetime import datetime
import json

class CrimePredictionPipeline:
    def __init__(self):
        self.setup_logging()
        self.data_ingestion = CrimeDataIngestion()
        self.data_processor = CrimeDataProcessor()
        self.predictor = CrimePredictionModel()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def run_pipeline(self):
        """Execute the complete prediction pipeline"""
        try:
            # Step 1: Download latest data
            self.logger.info("Downloading latest crime data...")
            data_file = self.data_ingestion.download_data()
            
            # Step 2: Process the data
            self.logger.info("Processing crime data...")
            raw_df = self.data_processor.spark.read.csv(data_file, header=True)
            processed_df = self.data_processor.process_raw_data(raw_df)
            
            # Step 3: Create zone profiles
            self.logger.info("Creating zone profiles...")
            zone_profiles = self.data_processor.create_zone_profiles(processed_df)
            
            # Step 4: Prepare features for prediction
            self.logger.info("Preparing features...")
            feature_df = self.predictor.prepare_features(processed_df)
            
            # Step 5: Train zone-specific models
            self.logger.info("Training zone-specific models...")
            zone_models = self.predictor.train_zone_models(feature_df)
            
            # Step 6: Generate predictions for next 8 days
            self.logger.info("Generating predictions...")
            predictions = self.predictor.predict_next_8_days(feature_df, zone_models)
            
            # Step 7: Save predictions
            self.save_predictions(predictions)
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
            
    def save_predictions(self, predictions):
        """Save predictions to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions/crime_predictions_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
            
        self.logger.info(f"Predictions saved to {filename}")

if __name__ == "__main__":
    pipeline = CrimePredictionPipeline()
    pipeline.run_pipeline() 