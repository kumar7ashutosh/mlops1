import logging
import os
from datetime import datetime

# Define the log directory and file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE) # This will create a 'logs' folder in your current working directory

# --- IMPORTANT CHANGE STARTS HERE ---
# Create the directory if it doesn't exist
# We need to get the directory part of the logs_path
log_dir = os.path.dirname(logs_path)
os.makedirs(log_dir, exist_ok=True)
# --- IMPORTANT CHANGE ENDS HERE ---

logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

# You might want to add a line to confirm logging is set up
# logging.info("Logging setup complete!")