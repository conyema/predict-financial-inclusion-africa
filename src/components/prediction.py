import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass


from src.exception import CustomException
# from src.logger import logging
from src.utils import load_object
from src.components.data_preprocessing import DataPreprocessor


# All configuration this components(class) requires
@dataclass
class PredictorConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_obj_path = os.path.join("artifacts", "label_encoder.pkl")
    base_model_path = os.path.join("artifacts", "base_model.pkl")
    optimized_model_path = os.path.join("artifacts", "optimized_model.pkl")


class Features:
    def __init__(
        self,
        **features
    ):

        self.features = features
        # self.location_type = features.location_type,
        # self.cellphone_access = features.cellphone_access,
        # self.household_size = features.household_size,
        # self.age = features.age,
        # self.gender = features.gender,
        # self.relationship_with_head = features.relationship_with_head,
        # self.marital_status = features.marital_status,
        # self.education_level = features.education_level,
        # self.job_type = features.job_type

    def convert_to_dataframe(self):
        "Converts data to a dataframe the preprocessor and model can use"

        try:
            data_dict = {
                "location_type": self.features["location_type"],
                "cellphone_access": self.features["cellphone_access"],
                "household_size": self.features["household_size"],
                "age_of_respondent": self.features["age"],
                "gender_of_respondent": self.features["gender"],
                "relationship_with_head": self.features["relationship_with_head"],
                "marital_status": self.features["marital_status"],
                "education_level": self.features["education_level"],
                "job_type": self.features["job_type"]
            }

            return pd.DataFrame(data_dict, index=[0])

        except Exception as e:
            raise CustomException(e, sys)


class Predictor:
    def __init__(self):
        self.config = PredictorConfig()

    def predict(self, data: pd.DataFrame):

        try:
            # Load artifacts
            base_model = load_object(self.config.base_model_path)
            opt_model = load_object(self.config.optimized_model_path)

            # Prepare data
            data_encoded = self.process_new_data(data)

            # Results from base model
            class_base, probability_base = self.get_class_and_probability(
                base_model, data_encoded)

            # Results from optimized model
            class_opt, probability_opt = self.get_class_and_probability(
                opt_model, data_encoded)

            # print("Class_base: ", class_base)
            # print("Class_opt: ", class_opt)
            # print("Probability1: ", probability_base)
            # print("Probability2: ", probability_opt)

            return class_base, class_opt, probability_base, probability_opt

        except Exception as e:
            raise CustomException(e, sys)

    def process_new_data(self, data):

        try:
            # Get encoder
            preprocessor = load_object(self.config.preprocessor_obj_path)

            # Transform data
            data_converted = DataPreprocessor().transform_data(data)

            # Encode data
            data_enc = preprocessor.transform(data_converted)

            return data_enc

        except Exception as e:
            raise CustomException(e, sys)

    def decode_class(self, prediction):
        ''' Returns the decoded class label from prediction '''

        try:
            # Get decoder
            label_encoder = load_object(self.config.label_encoder_obj_path)

            # Decode data
            class_label = label_encoder.inverse_transform(prediction)[0]

            return class_label

        except Exception as e:
            raise CustomException(e, sys)

    def get_class_and_probability(self, model, new_data):
        ''' Returns the label and probability of the predicted class'''

        try:
            prediction = model.predict(new_data)
            probabilities = model.predict_proba(new_data)
            class_label = self.decode_class(prediction)

            class_index = prediction[0]
            probability = probabilities[0][class_index]

            return class_label, probability

        except Exception as e:
            raise CustomException(e, sys)
