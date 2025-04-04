from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.sql.functions import *
from pyspark.sql.types import *
import logging

class CrimePredictionModel:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("LA Crime Prediction") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def prepare_features(self, df):
        """Prepare features for prediction"""
        # Convert date and time to features
        df = df.withColumn("hour", hour(col("DATE OCC"))) \
               .withColumn("day", dayofmonth(col("DATE OCC"))) \
               .withColumn("month", month(col("DATE OCC"))) \
               .withColumn("day_of_week", dayofweek(col("DATE OCC")))
        
        # Create crime type categories
        df = df.withColumn("crime_category", 
                          when(col("Part 1-2") == 1, "violent")
                          .when(col("Part 1-2") == 2, "property")
                          .otherwise("other"))
        
        # Feature columns for prediction
        feature_cols = ["hour", "day", "month", "day_of_week", "LAT", "LON", "AREA"]
        
        # Assemble features into vector
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df = assembler.transform(df)
        
        # Scale features
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        df = scaler.fit(df).transform(df)
        
        return df
        
    def train_zone_models(self, df):
        """Train separate models for each zone"""
        zones = df.select("AREA NAME").distinct().collect()
        zone_models = {}
        
        for zone in zones:
            zone_name = zone["AREA NAME"]
            zone_data = df.filter(col("AREA NAME") == zone_name)
            
            # Train classification model for crime type prediction
            rf_classifier = RandomForestClassifier(
                labelCol="Part 1-2",
                featuresCol="scaled_features",
                numTrees=100
            )
            
            # Train regression model for crime count prediction
            rf_regressor = RandomForestRegressor(
                featuresCol="scaled_features",
                labelCol="crime_count",
                numTrees=100
            )
            
            zone_models[zone_name] = {
                "classifier": rf_classifier.fit(zone_data),
                "regressor": rf_regressor.fit(zone_data)
            }
            
        return zone_models
        
    def predict_next_8_days(self, df, zone_models):
        """Generate predictions for next 8 days for each zone"""
        current_date = df.agg(max("DATE OCC")).collect()[0][0]
        predictions = []
        
        for i in range(1, 9):
            future_date = current_date + timedelta(days=i)
            
            # Create prediction data for each zone
            for zone_name, models in zone_models.items():
                # Generate features for prediction
                pred_data = self.spark.createDataFrame([
                    (future_date.hour, future_date.day, future_date.month,
                     future_date.dayofweek(), zone_name)
                ], ["hour", "day", "month", "day_of_week", "AREA NAME"])
                
                # Prepare features
                pred_data = self.prepare_features(pred_data)
                
                # Make predictions
                crime_type = models["classifier"].transform(pred_data)
                crime_count = models["regressor"].transform(pred_data)
                
                predictions.append({
                    "date": future_date,
                    "zone": zone_name,
                    "predicted_crime_type": crime_type.select("prediction").collect()[0][0],
                    "predicted_crime_count": crime_count.select("prediction").collect()[0][0]
                })
                
        return predictions 