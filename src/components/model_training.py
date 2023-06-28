import sys
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from dataclasses import dataclass


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier
# from sklearn.ensemble import r
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform

# import packages for random oversampling training data
# from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
# from imblearn.combine import SMOTETomek

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object
from src.components.data_preprocessing import DataPreprocessor


# All configuration this components(class) requires
@dataclass
class ModelTrainerConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_obj_path = os.path.join("artifacts", "label_encoder.pkl")
    base_model_path = os.path.join("artifacts", "base_model.pkl")
    optimized_model_path = os.path.join("artifacts", "optimized_model.pkl")
    report_dir = os.path.join("artifacts", "evaluations")
    processed_data_dir = os.path.join("artifacts", "processed")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_training(
            self,
            cleaned_data_path: str = "",
            metrics_priority: "list[str]" = [
                'mcc', 'auc_roc', 'f1', 'accuracy']
            # selection_metric: str= "",
            # selection_benchmark: float = 0.4
    ):

        try:
            # Use provided or configured data path
            data_dir = cleaned_data_path or self.config.processed_data_dir

            # Candidate models
            self.models = {
                "Logistic Regression": LogisticRegression(random_state=1),
                "K-Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=1),
                "Random Forest": RandomForestClassifier(random_state=1),
                # "XGB": XGBClassifier(),
                "GaussianNB": GaussianNB()
            }

            # Set metrics
            self.metrics = metrics_priority

            # Get and save datasets as instance attribute
            self.X_train, self.X_test, self.y_train, self.y_test = self.get_processed_data(
                data_dir)

            # logging.info(f"Data saved as instance attribute")

        except Exception as e:
            raise CustomException(e, sys)

    def train(
        self,
        save_best_model: bool = False,
        save_report: bool = False,
        # show_plots: bool = False,
        report_title: str = ""
    ):

        try:
            # evaluation_report, model_list = self.evaluate_models(
            # self.models, show_plots=show_plots)

            evaluation_report, model_list = self.evaluate_models(self.models)

            sorted_report = evaluation_report.sort_values(
                self.metrics, ascending=False)

            if save_report:
                # file_name = f"{report_title}.csv" or "base_model_report.csv"
                file_name = report_title or "base_model_report"

                # Make directory if it doesn't exist
                os.makedirs(self.config.report_dir, exist_ok=True)

                # Save sorted report as CSV file (on disk)
                sorted_report.to_csv(
                    os.path.join(self.config.report_dir, f"{file_name}.csv"),
                    index=False
                )

            if save_best_model:
                # evaluation_report.sort_values('MCC')['MCC'].max()
                # [name, acc, auc, f1, mcc] = evaluation_report[evaluation_report['MCC'] == evaluation_report['MCC'].max()].values[0]
                # model_index = evaluation_report[evaluation_report['mcc'] == evaluation_report['mcc'].max()].index

                model_info = sorted_report.head(1).values[0]
                model_index = sorted_report.head(1).index[0]

                best_model = model_list[model_index]

                # Save the best candidate model
                save_object(best_model, self.config.base_model_path)

                log_msg = f"Saved model:{model_info[0]} with performance: Accuracy - {model_info[1]}, AUC_ROC - {model_info[2]}, F1 - {model_info[3]},  MCC - {model_info[4]}"
                logging.info(log_msg)

        except Exception as e:
            raise CustomException(e, sys)

    def optimized_train(self, save_best_model: bool = False, save_report: bool = False, report_title: str = ""):

        try:
            # tol=1e-2   uniform(loc=0, scale=4)
            param_grid = {
                "Logistic Regression": {
                    # "penalty": ['l1', 'l2', None],
                    # "C": [x for x in np.linspace(0, 1)],
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "solver": ['lbfgs', 'liblinear', 'newton-cholesky', 'sag', 'saga']
                    # "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                    # "max_iter": [int(x) for x in np.linspace(start=80, stop=300, num=20)]
                },
                "K-Neighbors": {
                    "n_neighbors": [5, 6, 7, 8, 9]
                },
                "Decision Tree": {},
                "Random Forest": {
                    "n_estimators": [int(x) for x in np.linspace(start=80, stop=300, num=10)],
                    "bootstrap": [True, False]
                },
                "GaussianNB": {}
            }

            evaluation_report, model_list, best_params = self.evaluate_models(
                self.models, param_grid, hypertune=True)

            sorted_report = evaluation_report.sort_values(
                self.metrics, ascending=False)

            if save_report:
                # file_name = f"{report_title}.csv" or "base_model_report.csv"
                file_name = report_title or "optimized_model_report"

                # Make directory if it doesn't exist
                os.makedirs(self.config.report_dir, exist_ok=True)

                # Save sorted report as CSV file (on disk)
                sorted_report.to_csv(
                    os.path.join(self.config.report_dir, f"{file_name}.csv"),
                    index=False
                )

            if save_best_model:
                model_info = sorted_report.head(1).values[0]
                model_index = sorted_report.head(1).index[0]
                model_params = best_params[model_index]

                best_model = model_list[model_index]
                best_model.set_params(**model_params)

                # Save the best candidate model
                save_object(best_model, self.config.optimized_model_path)

                log_msg = f"Saved model:{model_info[0]} with performance: Accuracy - {model_info[1]}, AUC_ROC - {model_info[2]}, F1 - {model_info[3]},  MCC - {model_info[4]}"
                logging.info(log_msg)

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_models(self, models: dict, params: dict = {}, hypertune: bool = False,):

        try:
            # List(s) of performance metrics
            accuracy_scores: list = []
            auc_scores: list = []
            f1_scores: list = []
            mcc_scores: list = []
            best_params: list = []

            model_names = list(models.keys())
            model_list = list(models.values())

            for index, model in enumerate(model_list):
                if hypertune:
                    # Get params
                    param_dist = params[model_names[index]]

                    # Define evaluation
                    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                    cv = RepeatedStratifiedKFold(
                        n_splits=5, n_repeats=3, random_state=1)

                    # Define search
                    clf = RandomizedSearchCV(
                        estimator=model, param_distributions=param_dist, scoring='f1_weighted', cv=cv,  n_jobs=-1)

                    # clf = RandomizedSearchCV(
                    #     estimator=model, param_distributions=param_dist, scoring='matthews_corrcoef', cv=cv,  n_jobs=-1)

                    # Execute search for best hyperparameter combination
                    model = clf.fit(self.X_train, self.y_train)

                else:
                    model.fit(self.X_train, self.y_train)

                y_pred = model.predict(self.X_test)
                y_prob = model.predict_proba(self.X_test)[:, 1]

                # Get evaluation metrics
                accuracy_scores.append(
                    model.score(self.X_test, self.y_test)
                )

                auc_scores.append(
                    roc_auc_score(self.y_test, y_prob)
                )

                f1_scores.append(
                    f1_score(self.y_test, y_pred)
                )

                mcc_scores.append(
                    matthews_corrcoef(self.y_test, y_pred)
                )

                if hypertune:
                    best_params.append(
                        model.best_params_
                    )

                # Display evaluation plot(s)
                # if show_plots:
                #     self.display_plots(model, self.y_test, y_pred)

            # Create a Report "Table"
            evaluation_report = pd.DataFrame({
                'model name': model_names,
                'accuracy': accuracy_scores,
                'auc_roc': auc_scores,
                'f1': f1_scores,
                'mcc': mcc_scores
            })

            if hypertune:
                return evaluation_report, model_list, best_params

            return evaluation_report, model_list

        except Exception as e:
            raise CustomException(e, sys)

    # def optimized_train(self):

    #     try:
    #         # Define the parameter distributions
    #         params = {
    #             "Logistic Regression": {
    #                 "penalty": ['l1', 'l2', 'elasticnet', None],
    #                 "C": [x for x in np.linspace(0, 1)],
    #                 "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    #                 "max_iter": [int(x) for x in np.linspace(start=80, stop=300, num=20)]
    #             },
    #             "K-Neighbors": {
    #                 "n_neighbors": [5, 6, 7, 8, 9, 10, 11, 12]
    #             },
    #             "Decision Tree": {},
    #             "Random Forest": {
    #                 "n_estimators": [int(x) for x in np.linspace(start=80, stop=300, num=20)],
    #                 "bootstrap": [True, False]
    #             },
    #             "GaussianNB": {}
    #         }

    #         # # Define pipeline
    #         # steps = [('over', over_sampler), ('model', clf)]
    #         # clf_pipeline = Pipeline(steps=steps)

    #         # Find best classifier parameters
    #         # tuneModel(clf_pipeline, clf_params)

    #         # Define evaluation
    #         # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

    #         # # Define search
    #         # # clf = RandomizedSearchCV(n_iter=20, scoring='f1_micro')
    #         # clf = RandomizedSearchCV(
    #         #     estimator=model, param_distributions=params, scoring='f1', cv=cv,  n_jobs=-1)

    #         # # Execute search for best hyperparameter combination
    #         # result = clf.fit(X_train, y_train)

    #         # print('Best Score: %s' % result.best_score_)
    #         # print('Best Hyperparameters: %s' % result.best_params_)

    #     except:
    #         pass

    def get_processed_data(self, data_dir: str = ""):
        ''' Returns encoded train/test sets' features and target (X_train, X_test, y_train, y_test) '''

        try:
            # Use provided or configured data path
            data_dir = data_dir or self.config.processed_data_dir

            # Get clean and encoded train/test datasets
            X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
            X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
            y_train = pd.read_csv(os.path.join(
                data_dir, "y_train.csv")).squeeze()
            y_test = pd.read_csv(os.path.join(
                data_dir, "y_test.csv")).squeeze()

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)

    def sample_imbalanced_data(self):
        try:
            # Create an oversampler for the minority class
            # over_sampler = RandomOverSampler( random_state=1)
            # over_sampler = SMOTETomek(random_state=1)
            over_sampler = SMOTE(sampling_strategy='minority', random_state=1)

            # # fit and apply the transform
            self.X_train, self.y_train = over_sampler.fit_resample(
                self.X_train, self.y_train)

        except Exception as e:
            raise CustomException(e, sys)
