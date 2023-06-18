from src.components.data_ingestion import DataIngestion
from src.logger import logging


def main():
    # instantiate the ingestor
    data_ingestion = DataIngestion()

    # Get and Split the dataset
    data_ingestion.get_data()
    data_ingestion.split_data(test_size=0.2)


if __name__ == "__main__":
    main()
