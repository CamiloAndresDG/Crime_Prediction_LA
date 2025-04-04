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
        
    def process_raw_data(self, df):
        """Process raw crime data"""
        try:
            # Convert date and time columns
            df = df.withColumn("DATE OCC", to_timestamp("DATE OCC")) \
                   .withColumn("Date Rptd", to_timestamp("Date Rptd"))
            
            # Clean and standardize location data
            df = df.withColumn("LAT", col("LAT").cast("double")) \
                   .withColumn("LON", col("LON").cast("double")) \
                   .filter(col("LAT").isNotNull() & col("LON").isNotNull())
            
            # Create time-based features
            df = df.withColumn("hour", hour("DATE OCC")) \
                   .withColumn("day", dayofmonth("DATE OCC")) \
                   .withColumn("month", month("DATE OCC")) \
                   .withColumn("year", year("DATE OCC")) \
                   .withColumn("day_of_week", dayofweek("DATE OCC"))
            
            # Aggregate crime counts by location and time
            crime_counts = df.groupBy(
                "AREA NAME",
                "LAT",
                "LON",
                "year",
                "month",
                "day"
            ).agg(
                count("*").alias("crime_count"),
                collect_set("Crm Cd Desc").alias("crime_types")
            )
            
            return crime_counts
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise
            
    def create_zone_profiles(self, df):
        """Create profiles for each zone"""
        try:
            zone_profiles = df.groupBy("AREA NAME").agg(
                avg("crime_count").alias("avg_daily_crimes"),
                stddev("crime_count").alias("std_daily_crimes"),
                collect_set("crime_types").alias("common_crime_types")
            )
            
            return zone_profiles
            
        except Exception as e:
            self.logger.error(f"Error creating zone profiles: {str(e)}")
            raise 