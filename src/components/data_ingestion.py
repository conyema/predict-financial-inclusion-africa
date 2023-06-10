import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data .csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def get_data(self):
        logging.info("Data ingestion component/method")

        try:
            # Get the raw data
            df = pd.read_csv("notebook\data\Train.csv")
            logging.info("Dataset obtained as a dataframe")

            # Save data as CSV
            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,
                      header=True, index=False)
            logging.info("Raw data saved and data splitting initiated")

            # Split the raw data
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=1)

            # Save train and test data as CSV
            train_set.to_csv(
                self.ingestion_config.train_data_path, header=True, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            header=True, index=False)
            logging.info("Data ingestion complete")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.get_data()
