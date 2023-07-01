from src.components.data_ingestion import DataIngestion
from src.logger import logging


def main():
    logging.info(">>>>> data ingestion started <<<<<")

    # instantiate the ingestor
    data_ingestion = DataIngestion()

    # Get the dataset
    data_ingestion.get_data()

    logging.info(">>>>> data ingestion completed <<<<<")


if __name__ == "__main__":
    main()
