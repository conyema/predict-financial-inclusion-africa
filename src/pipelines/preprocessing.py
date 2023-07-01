from src.components.data_preprocessing import DataPreprocessor
from src.logger import logging


def main():
    cols_to_drop = ['uniqueid', 'country', 'year', 'bank_account']
    label = 'bank_account'

    logging.info(">>>>> data preprocessing started <<<<<")

    # Instantiate the processor
    preprocessor = DataPreprocessor()

    # Get cleaned, transformed and Split dataset
    X_train, X_test, y_train, y_test = preprocessor.initiate_preprocessing(
        label, cols_to_drop=cols_to_drop, test_size=0.2, use_data=True)

    # Save tranformers to disk
    preprocessor.create_preprocessor_objects(X_train, y_train)

    preprocessor.encode_training_data(
        X_train, X_test, y_train, y_test, use_data=True)

    # Save the processed/split and encoded data
    preprocessor.save_instance_data()

    logging.info(">>>>> data preprocessing completed <<<<<")


if __name__ == "__main__":
    main()
