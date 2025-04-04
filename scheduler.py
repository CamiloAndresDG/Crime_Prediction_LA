from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import logging
from src.pipeline import run_pipeline

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def schedule_pipeline():
    logger = setup_logging()
    scheduler = BlockingScheduler()
    
    # Schedule the pipeline to run every 8 days
    scheduler.add_job(
        run_pipeline,
        'interval',
        days=8,
        next_run_time=datetime.now()  # Run immediately on start
    )
    
    logger.info("Pipeline scheduler started")
    scheduler.start()

if __name__ == "__main__":
    schedule_pipeline() 