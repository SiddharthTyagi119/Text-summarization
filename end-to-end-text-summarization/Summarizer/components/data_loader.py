from Summarizer.components.news_dataset import NewsDataset
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from Summarizer.constant import *

# https://www.askpython.com/python/pytorch-lightning
# https://lightning.ai/docs/pytorch/stable/data/datamodule.html
class NewsDataLoader(pl.LightningDataModule):
    """
    NewsDataLoader class is a subclass of pl.LightningDataModule
    NewsDataLoader its inhiriate from pl.LightningDataModule
    It handle data loading and preprocessing for our prject dataset 
    """
    def __init__(self, train_file_path, val_file_path, tokenizer, batch_size,
                 columns_name, source_len=1024, target_len=128, corpus_size=500):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.batch_size = batch_size
        self.nrows = corpus_size
        self.columns_name = columns_name
        self.target_len = target_len
        self.source_len = source_len

    def prepare_data(self):
        """
        # Perform any initial setup or data downloading here

        """
        train_data = pd.read_csv(self.train_file_path,
                                 nrows=self.nrows/0.80, encoding='latin-1') 
        # 80% of the data we are going to read and same data we can use for the training part
        
        val_data = pd.read_csv(
            self.val_file_path, nrows=self.nrows/0.20, encoding='latin-1')
        
        train_data = train_data[self.columns_name]
        val_data = val_data[self.columns_name]

        self.train_data = train_data.dropna() # dropna method removes the rows that contains NULL values
        self.val_data = val_data.dropna() # dropna method removes the rows that contains NULL values

    def setup(self, stage=None):
        """
        # Load and split the dataset into train and validation

        X_train = select all rows (:) and the second-to-last column (-2).

        y_train =  select all rows (:) and the last column (-1).

        """
        X_train = self.train_data.iloc[:, -2].values 
        y_train = self.train_data.iloc[:, -1].values

        X_val = self.val_data.iloc[:, -2].values
        y_val = self.val_data.iloc[:, -1].values

        self.train_dataset = (X_train, y_train)
        self.val_dataset = (X_val, y_val)

    def train_dataloader(self):
        """
        # Return a DataLoader for the training set

        [0] indexing syntax specifies that you want to select the first element of the train_dataset

        [1] indexing syntax specifies that you want to select the second element of the train_dataset. 

        """
        train_data = NewsDataset(source_texts=self.train_dataset[0],
                                 target_texts=self.train_dataset[1],
                                 tokenizer=self.tokenizer,
                                 source_len=self.source_len,
                                 target_len=self.target_len
                                 )
        return DataLoader(train_data, self.batch_size, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        """
        # Return a DataLoader for the validation set
        """
        val_data = NewsDataset(source_texts=self.val_dataset[0],
                               target_texts=self.val_dataset[1],
                               tokenizer=self.tokenizer,
                               source_len=self.source_len,
                               target_len=self.target_len
                               )
        return DataLoader(val_data, self.batch_size, num_workers=NUM_WORKERS, pin_memory=True)