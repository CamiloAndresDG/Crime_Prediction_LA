from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import logging

class CrimeDataProcessor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("LA Crime Data Processing") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def read_data(self, input_path):
        """Read CSV data into Spark DataFrame"""
        try:
            df = self.spark.read.csv(
                input_path,
                header=True,
                inferSchema=True
            )
            self.logger.info(f"Successfully read data from {input_path}")
            return df
        except Exception as e:
            self.logger.error(f"Error reading data: {str(e)}")
            raise

    def process_data(self, df):
        """Process crime data for prediction"""
        try:
            # Convert date columns
            df = df.withColumn("DATE OCC", to_timestamp("DATE OCC"))
            
            # Group by location and date
            processed_df = df.groupBy(
                "AREA NAME",
                "LAT",
                "LON",
                date_trunc("day", col("DATE OCC")).alias("DATE")
            ).agg(
                count("*").alias("crime_count"),
                collect_set("Crm Cd Desc").alias("crime_types")
            )
            
            self.logger.info("Data processing completed successfully")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise

    def save_processed_data(self, df, output_path):
        """Save processed data"""
        try:
            df.write.mode("overwrite").parquet(output_path)
            self.logger.info(f"Data saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise

if __name__ == "__main__":
    processor = CrimeDataProcessor() 