import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object


# All configuration this components(class) requires
@dataclass
class DataPreprocessorConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_obj_path = os.path.join("artifacts", "label_encoder.pkl")
    root_dir = os.path.join("artifacts", "processed")
    data_path = os.path.join("artifacts", "data", "dataset.csv")


class DataPreprocessor:
    def __init__(self):
        self.config = DataPreprocessorConfig()
        # self.numerical_features: list[str] = []
        # self.categorical_features: list[str] = []
        # self.X_train:  any
        # self.X_test: any

    def remove_useless_categories(self, data):
        '''Drops instances with "Dont Know", "other", "Refuse to answer(RTA) inplace'''

        try:
            # Instances with "Dont Know", "other", "Refuse to answer(RTA)"
            rta_job = (data['job_type'] == 'Dont Know/Refuse to answer')
            rta_education = (data['education_level'] == 'Other/Dont know/RTA')
            rta_marital = (data['marital_status'] == 'Dont know')

            rta_conditions = rta_job | rta_education | rta_marital
            rta_indexes = set(data[rta_conditions].index)

            # Drop RTA instances
            data.drop(rta_indexes, inplace=True)

        except Exception as e:
            raise CustomException(e, sys)

    def find_outliers(self, data, drop: bool = False, exclude: "list[str]" = []):
        '''
        Returns and optionally drops rows/instances with outliers in continuous/numeric
        features from the provided dataframe(data)

        data: data in a dataframe
        drop: remove/drop outlier instances
        exclude: list of feature(s) whose outliers should be left

        '''

        try:
            # Collect unique indexes
            outliers_indexes: set[int] = set()

            # Get names of the Continuous features
            numerical_features: list[str] = data.select_dtypes(
                include='number').columns

            for col in numerical_features:

                if (exclude != []) & (col in exclude):
                    pass

                else:

                    feature = data[col]

                    # Compute the first and third quantiles and IQR of the feature
                    q1 = np.quantile(feature, 0.25)
                    q3 = np.quantile(feature, 0.75)
                    iqr = q3 - q1

                    # Calculate the lower and upper cutoffs for outliers
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr

                    # Subset feature to find outliers
                    outliers = feature[(feature < lower) | (feature > upper)]

                    # Update list of unique outliers
                    outliers_indexes.update(outliers.index)

                    logging.info(
                        f"Total outliers for {col} is {len(outliers)} at {round((len(outliers) / len(data) * 100), 2)}%")
                    # logging.info(outliers.index)

            if drop:
                data.drop(index=outliers_indexes, inplace=True)
                logging.info(
                    f"Removed {len(outliers_indexes)} rows/instance(s)")

        except Exception as e:
            raise CustomException(e, sys)

    def clean_data(self, data):

        try:
            # Remove useless categories
            self.remove_useless_categories(data)

            # Drop outlier instances
            self.find_outliers(data, drop=True)

        except Exception as e:
            raise CustomException(e, sys)

    def transform_data(self, data):
        '''Performs inplace type convertion/correction and lognormalization of numerical features'''

        try:
            # Get names of the Continuous features
            numerical_features: "list[str]" = data.select_dtypes(
                include='number').columns

            # Convert year type
            data['year'] = data['year'].astype('object')

            # Log normalize numerical features
            data[numerical_features] = np.log(
                data[numerical_features].astype("float"))

            return data

        except Exception as e:
            raise CustomException(e, sys)

    def split_trainset(self, X, y, test_size=0.3):
        '''
        Returns training and validation(test) arrays of features and target 
        '''

        try:
            logging.info("Data splitting initiated")

            # Split the raw data
            data_split: list = train_test_split(
                X, y, test_size=test_size, random_state=1, stratify=y)

            logging.info("Data splitting complete!!!")

            return(data_split)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_preprocessing(
            self,
            label: str,
            data_path: str = "",
            cols_to_drop: "list[str]" = [],
            test_size=0.3,
            use_data: bool = False
    ):
        '''
        Returns training/test sets split (X_train, X_test, y_train, y_test) from clean and transformed data

        use_data: Make the split data available (in memory) to this preprocessor instance

        '''

        try:
            logging.info("Data splitting initiated")

            # Provided path or configured path
            data_src = data_path or self.config.data_path

            # Read the raw data
            data = pd.read_csv(data_src)

            #  Normalize and convert type
            self.transform_data(data)

            # Clean the raw data
            self.clean_data(data)

            # Define the input features and target
            X = data.drop(columns=cols_to_drop)
            y = data[label]

            # Split the raw data
            X_train, X_test, y_train, y_test = self.split_trainset(
                X, y, test_size=test_size)

            if use_data:
                # Save train and test sets as instance attribute
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test

                logging.info(f"Split data saved as instance attribute")

            return(X_train, X_test, y_train, y_test)

        except Exception as e:
            raise CustomException(e, sys)

    def create_preprocessor_objects(self, X_train, y_train):
        '''Saves(on disk) an input preprocessor and a label encoder for independent and dependent features respectively from this instance's train set'''

        try:
            # Get names of the categorical and numerical features
            cat_features: "list[str]" = X_train.select_dtypes(
                exclude='number').columns
            num_features: "list[str]" = X_train.select_dtypes(
                include="number").columns
            # num_exists = num_features.empty == False

            num_steps = [("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())]

            cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")),
                         ("one_hot_encoder", OneHotEncoder(dtype=int, handle_unknown='ignore', sparse_output=False))]

            num_pipeline = Pipeline(num_steps)
            cat_pipeline = Pipeline(cat_steps)

            ct = ColumnTransformer(
                [
                    ("cat_preprocess", cat_pipeline, cat_features),
                    ("num_preprocess", num_pipeline, num_features)
                ]
            )

            # Fit the features
            preprocessor = ct.fit(X_train)

            # Fit the target encoder
            label_encoder = LabelEncoder().fit(y_train)

            # Save the transformers
            save_object(preprocessor, self.config.preprocessor_obj_path)
            save_object(label_encoder, self.config.label_encoder_obj_path)

        except Exception as e:
            raise CustomException(e, sys)

    def encode_training_data(self, X_train, X_test, y_train, y_test, use_data: bool = False):
        ''' 
        Returns encoded train/test sets X_train, X_test, y_train, y_test

        use_data: Make the encoded data available (in memory) to this preprocessor instance
        '''

        try:
            # Encoders
            label_encoder = load_object(self.config.label_encoder_obj_path)
            preprocessor = load_object(self.config.preprocessor_obj_path)

            # Encode datasets
            X_train_enc = preprocessor.transform(X_train)
            X_test_enc = preprocessor.transform(X_test)
            y_train_enc = label_encoder.transform(y_train)
            y_test_enc = label_encoder.transform(y_test)

            if use_data:
                # Save train and test sets as instance attribute
                self.X_train = X_train_enc
                self.X_test = X_test_enc
                self.y_train = y_train_enc
                self.y_test = y_test_enc

                logging.info(f"Data saved as instance attribute")

            return X_train_enc, X_test_enc, y_train_enc, y_test_enc

        except Exception as e:
            raise CustomException(e, sys)

    def save_instance_data(self):
        '''Saves(on disk) the processed/split data from this preprocessor instance as CSV files'''

        try:
            # Make directory if it doesn't exist
            os.makedirs(self.config.root_dir, exist_ok=True)

            # Save train and test sets as CSV
            pd.DataFrame(self.X_train).to_csv(os.path.join(
                self.config.root_dir, "X_train.csv"), index=False)

            pd.DataFrame(self.X_test).to_csv(os.path.join(
                self.config.root_dir, "X_test.csv"), index=False)

            pd.DataFrame(self.y_train).to_csv(os.path.join(
                self.config.root_dir, "y_train.csv"), index=False)

            pd.DataFrame(self.y_test).to_csv(os.path.join(
                self.config.root_dir, "y_test.csv"), index=False)

            logging.info(f"Data saved in {self.config.root_dir}")

        except Exception as e:
            raise CustomException(e, sys)

    def process_new_data(self, data):
        pass

        # try:
        #     # Get encoder
        #     preprocessor = load_object(self.config.preprocessor_obj_path)

        #     # Transform data
        #     data_trans = self.transform_data(data)

        #     # Encode data
        #     data_enc = preprocessor.transform(data_trans)

        #     return data_enc

        # except Exception as e:
        #     raise CustomException(e, sys)

    # def decode_class(self, data):
    #     ''' Returns the decoded class label from  '''

    #     try:
    #         # Get encoder
    #         label_encoder = load_object(self.config.label_encoder_obj_path)

    #         # Encode data
    #         class_label = label_encoder.inverse_transform(data)

    #         return class_label

    #     except Exception as e:
    #         raise CustomException(e, sys)
