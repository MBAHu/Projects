import sys,os

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException






class TraininingPipeline:
    def __init__(self) -> None:
        self.data_ingestion = DataIngestion()

        self.data_transformation = DataTransformation()

        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()

            (
                train_arr,
                test_arr,
                preprocessor_file_path,
            ) = self.data_transformation.initiate_data_transformation(
                train_path=train_path, test_path = test_path
            )

            r2_square = self.model_trainer.initiate_model_trainer(

                train_arr,
                test_arr,
                preprocessor_file_path= preprocessor_file_path,
            )
            print("training completed. Trainer model score : ", r2_square)

        except Exception as e:
            raise CustomException(e, sys)

    
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            return feature_store_file_path
            
        except Exception as e:
            raise CustomException(e,sys)
        


        
    
    
    def start_data_transformation(self, feature_store_file_path):
        try:
            data_transformation = DataTransformation(feature_store_file_path= feature_store_file_path)
            train_arr, test_arr,preprocessor_path = data_transformation.initiate_data_transformation()
            return train_arr, test_arr,preprocessor_path
            
        except Exception as e:
            raise CustomException(e,sys)


    def start_model_training(self, train_arr, test_arr):
        try:
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(
                train_arr, test_arr
            )
            return model_score
            
        except Exception as e:
            raise CustomException(e,sys)
        

    def run_pipeline(self):
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_arr, test_arr,preprocessor_path = self.start_data_transformation(feature_store_file_path)
            r2_square = self.start_model_training(train_arr, test_arr)
            
            
            print("training completed. Trained model score : ", r2_square)

        except Exception as e:
            raise CustomException(e, sys)
    
