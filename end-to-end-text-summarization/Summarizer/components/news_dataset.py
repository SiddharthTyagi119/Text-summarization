from torch.utils.data import Dataset
from Summarizer.exception import CustomException
import sys
import re

class NewsDataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """
    """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
    """
    #define source and target data related things  
    def __init__(self, source_texts, target_texts, tokenizer, source_len, target_len):
        try:
            self.source_texts = source_texts
            self.target_texts = target_texts
            self.tokenizer = tokenizer
            self.source_len = source_len
            self.target_len = target_len
        except Exception as e:
            raise CustomException(e, sys)
    
    def __len__(self):
        return len(self.target_texts) - 1
    
    """returns the length of target_texts"""
    
    """

    By subtracting 1 from the length, it define that the 
    length returned by this method is one less than the actual length of self.target_texts
    
    """
    #data cleaning 
    def __getitem__(self, idx):

        """return the input ids, attention masks and target ids"""

        whitespace_handler = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
        """ 
        the code removes leading and trailing whitespace, replaces multiple consecutive newlines 
        with a space, and replaces multiple consecutive whitespace characters with a single space. 
        """

        text = " ".join(str(self.source_texts[idx]).split())
        summary = " ".join(str(self.target_texts[idx]).split())
        """
        the code is removing extra whitespace from the text and summary strings 
        obtained from self.source_texts and self.target_texts, respectively, 
        at the specified index idx. This can be useful for cleaning up the strings 
        and ensuring consistent whitespace formatting.
        """
        #tokenize and embedding
        source = self.tokenizer.batch_encode_plus(
                                                [whitespace_handler(text)], # List of texts to be encoded
                                                max_length= self.source_len, # Maximum length of the encoded sequence
                                                padding='max_length', # Pad the sequences to the max_length
                                                truncation=True,  # Truncate sequences that exceed max_length
                                                return_attention_mask=True,  # Return attention masks for the encoded sequences
                                                add_special_tokens=True, # Add special tokens (e.g., [CLS], [SEP]) to the sequences
                                                return_tensors='pt') # Return PyTorch tensors
        
        """
        max_length: The maximum length of the encoded sequence. This parameter specifies the 
        desired maximum length for the tokenized sequences. Sequences longer than max_length 
        will be truncated, and shorter sequences will be padded.

        padding: Specifies how the sequences should be padded. In this case, 
        'max_length' is used to pad the sequences to the max_length specified earlier.

        truncation: Determines whether sequences that exceed the max_length should be truncated. 
        If set to True, the tokenizer will truncate sequences longer than max_length.

        return_attention_mask ( its conatin only 0 and 1) -> 
        Ex: My name is shivan kumar -> 1,2,3,4,5 0,0,0,0
        My name is Sanju and I'm a Data Scientist -> 1,2,3,4,5,6,7,8,9

        for the attention_mask: its tell us in which number our sttention should be given
        for first text: 1,1,1,1,1,0,0,0,0
        for second text: ,1,1,1,1,1,1,1,1

        """

        target = self.tokenizer.batch_encode_plus([whitespace_handler(summary)],
                                                max_length = self.target_len,
                                                padding='max_length',
                                                truncation=True,
                                                return_attention_mask=True,
                                                add_special_tokens=True,
                                                return_tensors='pt')
        
        labels = target['input_ids']
        labels[labels == 0] = -100
        """
        labels[labels == 0] = -100 assigns the value -100 to the selected elements in labels.
        and its  indicate that those specific tokens should be ignored or masked during training or evaluation.

        """
        #return index and attention mask
        return (source['input_ids'].squeeze(),
                source['attention_mask'].squeeze(),
                labels.squeeze(),
                target['attention_mask'].squeeze())
    
    """
    .squeeze() is a method that removes dimensions of size 1 from a tensor. 
    It is used here to remove any singleton dimensions from the input_ids tensor.
    """