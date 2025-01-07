from Summarizer.components.data_ingestion import DataIngestion
from Summarizer.components.model_evaluation import ModelEvaluation
from Summarizer.entity.config_entity import DataIngestionConfig
from Summarizer.components.model_trainer import ModelTrainer
from Summarizer.components.data_loader import NewsDataLoader
from Summarizer.components.model_finetuner import T5smallFinetuner
from Summarizer.constant import *
from Summarizer.entity.config_entity import DataIngestionConfig, ModelTrainerConfig, ModelEvaluationConfig
from Summarizer.components.model_trainer import ModelTrainerArtifacts
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#data_ingestion = DataIngestion(DataIngestionConfig())

#data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

#model_name = PRETRAINED_MODEL_NAME
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#t5small_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#train_data_path = data_ingestion_artifact.train_file_path
#val_data_path = data_ingestion_artifact.test_file_path

#dataloader = NewsDataLoader(train_file_path=train_data_path,
  #                         val_file_path= val_data_path,
 #                          tokenizer=tokenizer,
 #                          batch_size=BATCH_SIZE,
 ##                          columns_name=COLUMNS_NAME)
#
#dataloader.prepare_data()
#dataloader.setup()

#model = T5smallFinetuner(model=t5small_model, tokenizer=tokenizer)

#model_trainer = ModelTrainer(ModelTrainerConfig(), model, dataloader)

#model_trainer_artifacts = model_trainer.initiate_model_trainer()

trained_model_path=r'C:\\Users\\shiva\\Desktop\\Text_Summarization\\artifacts\\07_02_2023_17_42_58\\model_training_artifacts\\trained_model\\model.pt', 
model_loss=3.4012808799743652

model_trainer_artifacts = ModelTrainerArtifacts(trained_model_path=trained_model_path, model_loss=model_loss)

model_evaluation = ModelEvaluation(ModelEvaluationConfig(), model_trainer_artifacts=model_trainer_artifacts)

model_evaluation_artifacts = model_evaluation.initiate_evaluation()

