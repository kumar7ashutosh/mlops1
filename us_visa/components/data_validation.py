import sys
import pandas as pd
from pandas import DataFrame
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Initializes the DataValidation component.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Output reference of data ingestion artifact stage.
            data_validation_config (DataValidationConfig): Configuration for data validation.
        """
        try:
            logging.info(f"{'>>'*20} Data Validation log started. {'<<'*20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e,sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name: validate_number_of_columns
        Description: This method validates if the number of columns in the dataframe
                     matches the expected number of columns defined in the schema.

        Args:
            dataframe (DataFrame): The DataFrame to validate.

        Returns:
            bool: True if the number of columns matches, False otherwise.
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Number of columns validation status: [{status}]")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Method Name: is_column_exist
        Description: This method validates the existence of all expected numerical
                     and categorical columns in the dataframe.

        Args:
            df (DataFrame): The DataFrame to validate.

        Returns:
            bool: True if all required columns exist, False otherwise.
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            # Check for missing numerical columns
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            # Check for missing categorical columns
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            # Return False if any columns are missing, otherwise True
            return not (len(missing_categorical_columns) > 0 or len(missing_numerical_columns) > 0)
        except Exception as e:
            raise USvisaException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        """
        Method Name: read_data
        Description: Static method to read a CSV file into a DataFrame.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            DataFrame: The loaded DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name: initiate_data_validation
        Description: This method orchestrates the data validation process,
                     including column count and existence checks.

        Returns:
            DataValidationArtifact: An artifact summarizing the validation results.
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation process.")

            # Read training and testing data
            train_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            logging.info("Validating number of columns for training and testing dataframes.")
            # Validate number of columns in training DataFrame
            status_train_cols = self.validate_number_of_columns(dataframe=train_df)
            if not status_train_cols:
                validation_error_msg += "Columns are missing in training dataframe. "
            
            # Validate number of columns in testing DataFrame
            status_test_cols = self.validate_number_of_columns(dataframe=test_df)
            if not status_test_cols:
                validation_error_msg += "Columns are missing in test dataframe. "

            logging.info("Validating existence of all required columns for training and testing dataframes.")
            # Validate existence of columns in training DataFrame
            status_train_exist = self.is_column_exist(df=train_df)
            if not status_train_exist:
                validation_error_msg += "Required numerical/categorical columns are missing in training dataframe. "
            
            # Validate existence of columns in testing DataFrame
            status_test_exist = self.is_column_exist(df=test_df)
            if not status_test_exist:
                validation_error_msg += "Required numerical/categorical columns are missing in test dataframe. "

            # Determine overall validation status
            validation_status = len(validation_error_msg) == 0

            if validation_status:
                logging.info("All schema validations passed. Data is valid.")
                final_validation_message = "Data validation successful. No schema or column issues found."
            else:
                logging.info(f"Data validation failed. Errors: {validation_error_msg}")
                final_validation_message = f"Data validation failed due to: {validation_error_msg}"
            
            # Create DataValidationArtifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=final_validation_message,
                # drift_report_file_path is kept in the artifact as per its definition,
                # but it won't contain Evidently-generated content anymore.
                # You might need to adjust DataValidationArtifact if this path is no longer relevant.
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            logging.info(f"{'>>'*20} Data Validation log completed. {'<<'*20}")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e