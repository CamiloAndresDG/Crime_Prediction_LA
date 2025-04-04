import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import logging

class CrimeDataAnalyzer:
    def __init__(self, data_path):
        """
        Initialize the Crime Data Analyzer
        
        Args:
            data_path (str): Path to the crime data CSV file
        """
        self.setup_logging()
        self.logger.info("Loading crime data...")
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
        """Prepare and clean the data for analysis"""
        try:
            # Convert date columns
            self.df['DATE OCC'] = pd.to_datetime(self.df['DATE OCC'])
            self.df['Date Rptd'] = pd.to_datetime(self.df['Date Rptd'])
            
            # Extract time components
            self.df['Hour'] = self.df['TIME OCC'].apply(lambda x: int(str(x).zfill(4)[:2]))
            self.df['Month'] = self.df['DATE OCC'].dt.month
            self.df['Year'] = self.df['DATE OCC'].dt.year
            self.df['DayOfWeek'] = self.df['DATE OCC'].dt.day_name()
            
            self.logger.info("Data preparation completed successfully")
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise

    def create_crime_heatmap(self, save_path='visualizations/crime_heatmap.html'):
        """
        Create an interactive heatmap of crime locations
        
        Args:
            save_path (str): Path to save the heatmap HTML file
        """
        try:
            self.logger.info("Generating crime heatmap...")
            
            # Create base map centered on LA
            m = folium.Map(
                location=[34.0522, -118.2437],
                zoom_start=11,
                tiles='CartoDB positron'
            )
            
            # Prepare heatmap data
            heat_data = self.df[['LAT', 'LON']].dropna().values.tolist()
            HeatMap(heat_data).add_to(m)
            
            # Save the map
            m.save(save_path)
            self.logger.info(f"Heatmap saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating heatmap: {str(e)}")
            raise

    def plot_crime_trends(self, save_path='visualizations/crime_trends.png'):
        """
        Plot various crime trends over time
        
        Args:
            save_path (str): Path to save the visualization
        """
        try:
            self.logger.info("Generating crime trends visualization...")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            
            # Daily crime counts
            daily_crimes = self.df.groupby('DATE OCC').size()
            ax1.plot(daily_crimes.index, daily_crimes.values)
            ax1.set_title('Daily Crime Incidents')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Number of Crimes')
            ax1.tick_params(axis='x', rotation=45)
            
            # Crime by hour
            hourly_crimes = self.df['Hour'].value_counts().sort_index()
            ax2.bar(hourly_crimes.index, hourly_crimes.values)
            ax2.set_title('Crime Incidents by Hour')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Number of Crimes')
            
            # Crime by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_crimes = self.df['DayOfWeek'].value_counts()
            weekly_crimes = weekly_crimes.reindex(day_order)
            ax3.bar(weekly_crimes.index, weekly_crimes.values)
            ax3.set_title('Crime Incidents by Day of Week')
            ax3.tick_params(axis='x', rotation=45)
            
            # Top crime types
            top_crimes = self.df['Crm Cd Desc'].value_counts().head(10)
            ax4.barh(top_crimes.index, top_crimes.values)
            ax4.set_title('Top 10 Crime Types')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"Crime trends visualization saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting crime trends: {str(e)}")
            raise

    def create_area_analysis(self, save_path='visualizations/area_analysis.html'):
        """
        Create interactive area-based crime analysis
        
        Args:
            save_path (str): Path to save the visualization
        """
        try:
            self.logger.info("Generating area-based analysis...")
            
            # Crime counts by area
            area_crimes = self.df.groupby('AREA NAME').size().reset_index(name='count')
            
            # Create interactive bar chart
            fig = px.bar(
                area_crimes,
                x='AREA NAME',
                y='count',
                title='Crime Incidents by Area',
                color='count',
                color_continuous_scale='Viridis'
            )
            
            fig.write_html(save_path)
            self.logger.info(f"Area analysis saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating area analysis: {str(e)}")
            raise

    def generate_crime_patterns(self, save_path='visualizations/crime_patterns.html'):
        """
        Generate interactive visualization of crime patterns
        
        Args:
            save_path (str): Path to save the visualization
        """
        try:
            self.logger.info("Generating crime patterns visualization...")
            
            # Create time-based patterns
            time_patterns = pd.crosstab(self.df['Hour'], self.df['DayOfWeek'])
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=time_patterns.values,
                x=time_patterns.columns,
                y=time_patterns.index,
                colorscale='Viridis'
            ))
            
            fig.update_layout(
                title='Crime Patterns by Hour and Day',
                xaxis_title='Day of Week',
                yaxis_title='Hour of Day'
            )
            
            fig.write_html(save_path)
            self.logger.info(f"Crime patterns visualization saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating crime patterns: {str(e)}")
            raise

    def create_statistical_summary(self, save_path='visualizations/statistical_summary.txt'):
        """
        Generate statistical summary of crime data
        
        Args:
            save_path (str): Path to save the summary
        """
        try:
            self.logger.info("Generating statistical summary...")
            
            with open(save_path, 'w') as f:
                # Overall statistics
                f.write("=== LA Crime Data Statistical Summary ===\n\n")
                f.write(f"Total number of incidents: {len(self.df):,}\n")
                f.write(f"Date range: {self.df['DATE OCC'].min()} to {self.df['DATE OCC'].max()}\n\n")
                
                # Crime type distribution
                f.write("Top 10 Crime Types:\n")
                crime_types = self.df['Crm Cd Desc'].value_counts()
                for crime_type, count in crime_types.head(10).items():
                    f.write(f"- {crime_type}: {count:,}\n")
                
                # Area statistics
                f.write("\nCrime by Area:\n")
                area_stats = self.df.groupby('AREA NAME').size()
                for area, count in area_stats.items():
                    f.write(f"- {area}: {count:,}\n")
                
                # Time-based statistics
                f.write("\nPeak Crime Hours:\n")
                hour_stats = self.df['Hour'].value_counts().sort_index()
                peak_hour = hour_stats.idxmax()
                f.write(f"Most active hour: {peak_hour:02d}:00 ({hour_stats[peak_hour]:,} incidents)\n")
                
            self.logger.info(f"Statistical summary saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating statistical summary: {str(e)}")
            raise

def main():
    """Main function to run the analysis"""
    try:
        # Initialize analyzer
        analyzer = CrimeDataAnalyzer("data/raw/crime_data.csv")
        
        # Create output directories
        import os
        os.makedirs('visualizations', exist_ok=True)
        
        # Generate all visualizations and analyses
        analyzer.create_crime_heatmap()
        analyzer.plot_crime_trends()
        analyzer.create_area_analysis()
        analyzer.generate_crime_patterns()
        analyzer.create_statistical_summary()
        
        print("Analysis completed successfully. Check the 'visualizations' directory for results.")
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 