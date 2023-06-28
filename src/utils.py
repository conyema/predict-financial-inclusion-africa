import os
import sys
from pickle import dump, load

from src.exception import CustomException


def save_object(obj, file_path):
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # save the object
        with open(file_path, 'wb') as file:
            dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        
        # file = open(file_path, "rb")
        # return load(file)

        with open(file_path, 'rb') as file:
            return load(file)


    except Exception as e:
        raise CustomException(e, sys)
