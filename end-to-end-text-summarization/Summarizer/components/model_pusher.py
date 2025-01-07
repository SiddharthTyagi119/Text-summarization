from Summarizer.entity.artifact_entity import ModelEvaluationArtifacts, ModelPusherArtifacts
from Summarizer.constant import TRAINED_MODEL_NAME, MODEL_PUSHER_DIR
from Summarizer.logger import logging
from Summarizer.exception import CustomException
import sys
import shutil


class ModelPusher:
    def __init__(self, model_evaluation_artifacts: ModelEvaluationArtifacts):
        self.model_evaluation_artifacts = model_evaluation_artifacts

    def initiate_model_pusher(self):
        try:
            logging.info("Initiating model pusher component")
            if self.model_evaluation_artifacts.saved_model_path is None: #Saved model directory is empty 
                # Artifact model
                trained_model_path = self.model_evaluation_artifacts.trained_model_path
                # Saved Model 
                saved_model_path = self.model_evaluation_artifacts.saved_model_path
                
                logging.info(f"Artifact model  : {trained_model_path}")
                
                logging.info(f"Saved Model : {saved_model_path}")
                
                
                shutil.copy(trained_model_path, saved_model_path)
                message = "Model Pusher pushed the current Trained model to Saved_Model_directory"
                response = {"is model pushed": True, "model_path": saved_model_path ,"message" : message}
                logging.info(response)
              # 2 models we have in our trained_model_dir  
              # again we are going to train our model -> new model -
              # total 3 model -> 87% -> 94% -> new model
              # current model we have in our saved_model_dir -> replace with new model becuase we are getting higer accurqacy model
            else:
                if self.model_evaluation_artifacts.is_model_accepted:
                    # Artifact model
                    trained_model_path = self.model_evaluation_artifacts.trained_model_path
                    # Saved Model 
                    saved_model_path = self.model_evaluation_artifacts.saved_model_path
                    
                    logging.info(f"Artifact model  : {trained_model_path}")
                
                    logging.info(f"Saved Model : {saved_model_path}")
                    
                    # Copying model from Artifact to saved_model_directory
                    shutil.copy(trained_model_path, saved_model_path)

                    message = "Model Pusher pushed the current Trained model to Saved_Model_directory"
                    response = {"is model pushed": True, "model_path": saved_model_path ,"message" : message}
                    logging.info(response)
                else:
                    saved_model_path = self.model_evaluation_artifacts.saved_model_path
                    message = "Current Trained Model is not accepted as saved_model has better loss"
                    response = {"is model pushed": False, "Saved_Model":saved_model_path,"message" : message}
                    logging.info(response)
            
            model_pusher_artifacts = ModelPusherArtifacts(response=response)
            return model_pusher_artifacts

        except Exception as e:
            raise CustomException(e, sys)
        