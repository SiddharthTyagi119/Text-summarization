import os, sys
from dataclasses import dataclass
from datetime import datetime
from Summarizer.constant import *

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# artifact | data_ingestion_timestap

#timestamp for the output generated dir 
@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = SOURCE_DIR_NAME
    artifact_dir: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    download_url: str = DOWNLOAD_URL
    file_name: str = METADATA_FILE_NAME
    # Mater Data Ingestion directory 
    data_ingestion_artifacts: str = os.path.join(ROOT_DIR, training_pipeline_config.artifact_dir, DATA_INGESTION_ARTIFACTS_DIR)
    
    # Data Ingestion Directory 
    raw_data_dir: str = os.path.join(ROOT_DIR, data_ingestion_artifacts, RAW_DATA_DIR_NAME)
    ingested_data: str = os.path.join(raw_data_dir,'ingested_data')
    
    
    # Split train and TEst Data
    train_file_path: str = os.path.join(data_ingestion_artifacts, DATA_INGESTION_TRAIN_DIR, TRAIN_FILE_NAME)
    test_file_path: str = os.path.join(data_ingestion_artifacts, DATA_INGESTION_TEST_DIR, TEST_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    model_trainer_artifacts_dir: str = os.path.join(ROOT_DIR, training_pipeline_config.artifact_dir, MODEL_TRAINING_ARTIFACTS_DIR)
    trained_model_dir: str = os.path.join(model_trainer_artifacts_dir, 'trained_model', TRAINED_MODEL_NAME)
    epochs: int = EPOCHS
    checkpoint_dir: str = os.path.join(model_trainer_artifacts_dir, CHECKPOINT_DIR)
    checkpoint_fname: str = 'best_checkpoint'

@dataclass
class ModelEvaluationConfig:
    saved_model_directory: str = SAVED_MODEL_DIRECTORY
    model_evaluation_artifacts_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR)
    in_channels: int = IN_CHANNELS
    base_loss: float = BASE_LOSS

@dataclass
class PredictionPipelineConfig:
    input_max_length: int = 400
    output_max_length: int = 100
    num_beams: int = 4
    base_model_name: str = PRETRAINED_MODEL_NAME
    prediction_artifact_dir = os.path.join(ROOT_DIR, PREDICTION_PIPELINE_DIR_NAME)
    model_download_path = os.path.join(prediction_artifact_dir, PREDICTION_MODEL_DIR_NAME)