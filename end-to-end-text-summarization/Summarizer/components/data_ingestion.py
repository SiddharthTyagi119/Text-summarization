import os
import sys
import requests
import zipfile
import pandas as pd
import urllib
import shutil
from dateutil import parser
from sklearn.model_selection import StratifiedShuffleSplit
from Summarizer.logger import logging
from Summarizer.exception import CustomException
from Summarizer.entity.config_entity import DataIngestionConfig
from Summarizer.entity.artifact_entity import DataIngestionArtifacts
from Summarizer.constant import *


class DataIngestion:
    """Ingest the data to the pipeline."""
#calling data ingestion config from config_entity.
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)
        
# download data
# split our complete data into train and test
# initiate our data ingestion part

    def download_data(self) -> str:
        try:
            # Raw Data Directory Path
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            # Make Raw data Directory
            os.makedirs(raw_data_dir, exist_ok=True)

            file_name = METADATA_FILE_NAME

            raw_file_path = os.path.join(raw_data_dir, file_name)

            # Download URL
            download_url = self.data_ingestion_config.download_url + "?raw=true"

            # Downloading the zip file
            logging.info(f"Downloading file from URL: {download_url}")

            urllib.request.urlretrieve(download_url, os.path.join(raw_data_dir, "data.zip"))

            logging.info("File downloaded successfully.")

            # Extracting the zip file
            with zipfile.ZipFile(os.path.join(raw_data_dir, "data.zip"), "r") as zip_ref:
                zip_ref.extractall(raw_data_dir)
            logging.info("Zip file extracted successfully.")

            # Delete the downloaded zip file
            os.remove(os.path.join(raw_data_dir, "data.zip"))
            
            # Extracting name of the csv file extracted 
            # Extracted CSV file path (assuming it has a .csv extension)
            csv_file_path = None

            # Get the list of files in the raw data directory
            file_list = os.listdir(raw_data_dir)

            # Search for the CSV file
            for file_name in file_list:
                if file_name.endswith(".csv"):
                    csv_file_path = os.path.join(raw_data_dir, file_name)
                    break
            # Print the name of the CSV file
            if csv_file_path is not None:
                csv_file_name = os.path.basename(csv_file_path)
                print("CSV file name:", csv_file_name)
                
            raw_file_path = os.path.join(raw_data_dir, csv_file_name)
            
            
            # copy the the extracted csv from raw_data_dir ---> ingested Data 
            ingest_file_path=os.path.join(self.data_ingestion_config.ingested_data)
            os.makedirs(ingest_file_path,exist_ok=True)
            
            
            # Copy the extracted CSV file
            shutil.copy2(raw_file_path, ingest_file_path)
            
            
            

            logging.info(f"File: {ingest_file_path} has been downloaded and extracted successfully.")
            return ingest_file_path

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def train_test_split(self,data_ingested_file_path) -> None:
        try:
            for file in os.listdir(data_ingested_file_path):
                if file.endswith('.csv'):
                    csv_file_path = os.path.join(data_ingested_file_path, file)

                    logging.info(f"Reading the CSV at {csv_file_path}")
                    
                    raw_data = pd.read_csv(csv_file_path, encoding='latin-1')
                    

            # instantiate shuffle split function
            split = StratifiedShuffleSplit(
                n_splits=5, train_size=0.85, test_size=0.15, random_state=42)
            
            # convert date(string) column to month and year format for stratified shuffle split
            for i in range(len(raw_data['date'])):
                raw_data['date'][i] = parser.parse(raw_data['date'][i]).strftime("%m-%Y")

            # strftime is a method used to format a datetime object into a string representation based on a specified format
            
            # stratified split using 
            for train_index, test_index in split.split(raw_data, raw_data['date']):
                strat_train_set = raw_data.loc[train_index]
                strat_test_set = raw_data.loc[test_index]
            
            # save train and test to specified path
            train_file_path = self.data_ingestion_config.train_file_path
            test_file_path = self.data_ingestion_config.test_file_path

            # create train and test directory if it doesn't exist
            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

            # save train and test set
            strat_train_set.to_csv(train_file_path, index=False)
            strat_test_set.to_csv(test_file_path, index=False)

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Starting data ingestion component...")
        try:
            data_ingested_file_path=self.download_data()

            self.train_test_split(data_ingested_file_path=data_ingested_file_path)
            
            data_ingestion_artifacts = DataIngestionArtifacts(raw_file_dir=self.data_ingestion_config.raw_data_dir,
                                                             train_file_path= self.data_ingestion_config.train_file_path,
                                                             test_file_path= self.data_ingestion_config.test_file_path
                                                             )

            logging.info(f"Data ingestion artifact is generated {data_ingestion_artifacts}")
            
            logging.info("Data ingestion completed successfully")
         
            
            return data_ingestion_artifacts
        
        except Exception as e:
            logging.error(
                "Error in Data Ingestion component! Check above logs")
            raise CustomException(e, sys)



        