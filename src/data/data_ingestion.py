import pandas as pd
import requests
from datetime import datetime
import os
import logging

class CrimeDataIngestion:
    def __init__(self):
        self.url = "https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD"
        self.data_dir = "data/raw"
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        self.logger.info(f"Created directory: {self.data_dir}")
        
    def download_data(self):
        """Download crime data from LA City Data"""
        try:
            self.create_directories()
            
            self.logger.info("Starting data download...")
            response = requests.get(self.url)
            response.raise_for_status()
            
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_dir}/crime_data_{current_time}.csv"
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Data successfully downloaded to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error downloading data: {str(e)}")
            raise

if __name__ == "__main__":
    ingestion = CrimeDataIngestion()
    ingestion.download_data() 