# 🚓 LA Crime Prediction System

![LA Crime Prediction](machine-learning.png)

A sophisticated machine learning pipeline that predicts crime patterns in Los Angeles using PySpark and advanced analytics. The system processes historical crime data to generate 8-day forecasts for different zones across the city.

## 🎯 Features

- **Real-time Data Integration**: Automated data ingestion from LA's official crime database
- **Zone-Specific Predictions**: Customized models for each LAPD district
- **Multi-Model Approach**: 
  - Crime type classification
  - Crime count prediction
  - Location-based pattern recognition
- **8-Day Forecasting**: Rolling predictions updated every 8 days
- **Scalable Processing**: Built with PySpark for handling large-scale data
- **Advanced Feature Engineering**: Temporal and spatial feature extraction

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/yourusername/Crime_Prediction_LA.git
cd Crime_Prediction_LA

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Create necessary directories
mkdir -p data/raw predictions

# Run the main pipeline
python src/pipeline.py
```

## 📊 Project Structure 
Crime_Prediction_LA/
├── src/
│ ├── data/
│ │ └── data_ingestion.py # Data download and storage
│ ├── processing/
│ │ └── data_processor.py # PySpark data processing
│ ├── models/
│ │ └── crime_predictor.py # Prediction models
│ └── pipeline.py # Main pipeline orchestrator
├── data/
│ └── raw/ # Raw data storage
├── predictions/ # Prediction outputs
├── notebooks/ # Jupyter notebooks
├── tests/ # Unit tests
├── requirements.txt # Project dependencies
└── README.md # Project documentation