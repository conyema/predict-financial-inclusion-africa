import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


# All configuration this components(class) requires
@dataclass
class DataIngestionConfig:
    # train_data_path: str = os.path.join("artifacts", "data", "train.csv")
    root_dir: str = os.path.join("artifacts", "data")
    source_data_path: str = "notebook/data/Train.csv"


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def get_data(self):

        try:
            # Get the raw data
            data = pd.read_csv(self.config.source_data_path)

            logging.info("Dataset obtained as a dataframe")

            # Create directory if it doesn't exist
            os.makedirs(self.config.root_dir, exist_ok=True)

            # Save data as CSV
            data.to_csv(
                os.path.join(self.config.root_dir, "dataset.csv"),
                header=True,
                index=False
            )

            logging.info(f"Raw dataset saved in {self.config.root_dir}")

            return (
                self.config.root_dir
            )

        except Exception as e:
            raise CustomException(e, sys)
