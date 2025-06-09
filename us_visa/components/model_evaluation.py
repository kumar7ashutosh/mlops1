from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from us_visa.exception import USvisaException
from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR
from us_visa.logger import logging
import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass

# Removed: from us_visa.entity.s3_estimator import USvisaEstimator
from us_visa.entity.estimator import USvisaModel
from us_visa.entity.estimator import TargetValueMapping
from us_visa.utils.main_utils import load_object # Added to load trained model


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            # Placeholder for best model if you want to load it from a local path
            # self.best_model_local_path = self.model_eval_config.best_model_local_path 
        except Exception as e:
            raise USvisaException(e, sys) from e

    # Removed the get_best_model method as it was dependent on S3Estimator

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate the trained model
                        and determine if it should be accepted (e.g., compared to a previous "best" model).

        Output      :   Returns an EvaluateModelResponse object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Starting model evaluation process.")

            # Load test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']

            # Separate features and target
            x = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]

            # Replace target values with numerical mapping
            target_mapping = TargetValueMapping()._asdict()
            y = y.replace(target_mapping)

            # Load the newly trained model
            trained_model_obj: USvisaModel = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )

            # Get f1 score for the newly trained model
            y_pred_trained_model = trained_model_obj.predict(x)
            trained_model_f1_score = f1_score(y, y_pred_trained_model)
            logging.info(f"Trained model F1 score: {trained_model_f1_score}")

            best_model_f1_score = None
            is_model_accepted = False
            difference = 0.0

            # You can add logic here to load a "best" model from a *local* path
            # For now, without S3, we'll assume the first model trained is the "best"
            # unless a local best model path is configured and model exists there.
            
            # Example: If you want to load a "production" model from a local path
            # if os.path.exists(self.model_eval_config.best_model_local_path):
            #     best_model_obj = load_object(file_path=self.model_eval_config.best_model_local_path)
            #     y_hat_best_model = best_model_obj.predict(x)
            #     best_model_f1_score = f1_score(y, y_hat_best_model)
            #     logging.info(f"Production model F1 score from local path: {best_model_f1_score}")
            #     is_model_accepted = trained_model_f1_score > best_model_f1_score
            #     difference = trained_model_f1_score - best_model_f1_score
            # else:
            #     logging.info("No existing best model found locally. New trained model is considered best.")
            #     is_model_accepted = True # Accept the first trained model if no other exists
            #     best_model_f1_score = trained_model_f1_score # For reporting consistency
            #     difference = 0.0

            # Simplification: If no external "best model" comparison is defined,
            # the trained model is always accepted and its F1 score is the "best".
            is_model_accepted = True
            best_model_f1_score = trained_model_f1_score
            difference = 0.0
            logging.info("No external best model comparison configured. Trained model is accepted by default.")


            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=difference
            )
            logging.info(f"Model evaluation result: {result}")
            return result

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation

        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Initiating model evaluation.")
            evaluate_model_response = self.evaluate_model()

            # s3_model_path is removed as S3 is not used
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                # s3_model_path=None, # No S3 path
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e