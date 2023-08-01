import logging
import os
import sys
from datetime import datetime


# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
# os.makedirs(logs_path, exist_ok=True)

# LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# logging.basicConfig(
#     filename=LOG_FILE_PATH,
#     format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO
# )

log_format = "[%(asctime)s]: %(name)s - %(levelname)s - %(module)s - %(message)s"
log_dir = os.path.join(os.getcwd(), "logs")
log_filepath = os.path.join(log_dir, "fia-prediction.log")

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    format=log_format,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

# if __name__ == "__main__":
#     logging.info("Logging has started")
