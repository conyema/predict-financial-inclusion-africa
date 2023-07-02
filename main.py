from src.pipelines import ingestion, preprocessing, training
from src.logger import logging


def main():
    # Get data
    ingestion.main()

    # Preprocess data
    preprocessing.main()

    # Train/build model
    training.main()



if __name__ == "__main__":
    main()


