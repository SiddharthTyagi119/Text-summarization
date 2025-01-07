import os, sys

ARTIFACTS_DIR: str = "artifacts"
SOURCE_DIR_NAME: str = 'Summarizer'

# Root Directory 
ROOT_DIR=os.getcwd()

# Config File path 
CONFIG_DIR='config'
CONFIG_FILE_NAME='config.yaml'
CONFIG_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)


DOWNLOAD_URL="https://github.com/Shivan118/Main-Branching/blob/main/news_summary.zip"


#Databse 
DATABASE_NAME='text_data'
COLLECTION_NAME='data'

# common files
METADATA_DIR = "metadata"
METADATA_FILE_NAME: str = "metadata.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# constants related to data ingestion
DATA_INGESTION_ARTIFACTS_DIR: str = "data_ingestion_artifacts"
RAW_DATA_DIR_NAME: str = "raw_data"
DATA_INGESTION_TRAIN_DIR: str = "train"
DATA_INGESTION_TEST_DIR: str = "test"


# constants related to model training
MODEL_TRAINING_ARTIFACTS_DIR: str = "model_training_artifacts"
TRAINED_MODEL_NAME: str = 'model.pt'
CHECKPOINT_DIR: str = 'checkpoint'
LEARNING_RATE: float = 2e-5
EPOCHS: int = 1
BATCH_SIZE = 4
NUM_WORKERS = 0


# constants related to pipeline
COLUMNS_NAME =  ['text', 'ctext']
PRETRAINED_MODEL_NAME = "t5-small"

# constants related to model evaluation
SAVED_MODEL_DIRECTORY=os.path.join(ROOT_DIR,'Saved_model')
MODEL_EVALUATION_DIR: str = "model_evaluation"
IN_CHANNELS: int = 1
BASE_LOSS: float = 10


# constants related to model pusher
MODEL_PUSHER_DIR: str = "model_pusher"


# constants related to prediction
PREDICTION_PIPELINE_DIR_NAME = "prediction_artifacts"
PREDICTION_MODEL_DIR_NAME = "prediction_model"