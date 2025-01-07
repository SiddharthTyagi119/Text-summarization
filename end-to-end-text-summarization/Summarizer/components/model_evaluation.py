import os
import sys
import torch
import numpy as np
from Summarizer.constant import *
from Summarizer.entity.config_entity import ModelEvaluationConfig
from Summarizer.entity.artifact_entity import ModelTrainerArtifacts, ModelEvaluationArtifacts
from Summarizer.logger import logging
from Summarizer.exception import CustomException

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.trainer_artifacts = model_trainer_artifacts

    def get_saved_model(self):
        """
        we are going to save best model here.
        """
        try:
            saved_model_directory = self.model_evaluation_config.saved_model_directory
            os.makedirs(saved_model_directory,exist_ok=True)

            for file in os.listdir(saved_model_directory):
                if file.endswith(".pt"):
                    saved_model_path = os.path.join(saved_model_directory, file)
                    logging.info(f"Best model found in {saved_model_path}")
                    return saved_model_path
                else:
                    saved_model_path = saved_model_directory
                    saved_model_path=None
                    logging.info(
                        "Model is not available in saved_model_directory")
                    return saved_model_path
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def evaluate_model(self):
        try:
            saved_model_path = self.get_saved_model()
            if saved_model_path is not None:
                # load back the model
                state_dict = torch.load(saved_model_path, map_location='cpu')
                loss = state_dict['loss']
                logging.info(f"Saved Model Validation loss : {loss}")
                logging.info(
                    f"Locally trained loss is {self.trainer_artifacts.model_loss}")
                saved_model_loss = loss
            else:
                logging.info(
                    "Model is not found in Saved_Model_Directory, So couldn't evaluate")
                saved_model_loss = None
            return saved_model_loss,saved_model_path
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_evaluation(self):
        try:
            # Getting model loss of model in the saved model directory 
            saved_model_loss,saved_model_path = self.evaluate_model()
            
            # saved_model_loss ----> Temporary_model_loss -> np.inf is infinity
            tmp_best_model_loss = np.inf if saved_model_loss is None else saved_model_loss
            
            # Artifact_model --->Model_loss
            trained_model_loss = self.trainer_artifacts.model_loss
            
            if saved_model_path is None:
                
                saved_model_path=os.path.join(self.model_evaluation_config.saved_model_directory,TRAINED_MODEL_NAME)
                
                logging.info(f" Saved Model direcodry is empty , Saved model path :{saved_model_path}")
                
            trained_model_path=self.trainer_artifacts.trained_model_path
            # Evaluation of:  Artifact model <-----> Saved_Model, Saved_Model<-----> base_loss 
            evaluation_response = tmp_best_model_loss > trained_model_loss and trained_model_loss < self.model_evaluation_config.base_loss
            model_evaluation_artifacts = ModelEvaluationArtifacts(saved_model_loss=tmp_best_model_loss,
                                                                  is_model_accepted=evaluation_response,
                                                                  trained_model_path=trained_model_path,
                                                                  saved_model_path=saved_model_path
                                                                  )
            
            # Temporary best model loss
            logging.info(
                f"Model evaluation completed! Artifacts: {model_evaluation_artifacts}")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)