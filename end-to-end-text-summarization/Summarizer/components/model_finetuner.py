import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
import torch
from Summarizer.exception import CustomException
import sys
from Summarizer.constant import LEARNING_RATE

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/T5/Fine_tune_CodeT5_for_generating_docstrings_from_Ruby_code.ipynb
# https://www.kaggle.com/code/deekoul/summarization-with-t5-pytorchlightning

class T5smallFinetuner(pl.LightningModule):
    def __init__(self, model, tokenizer, learning_rate=LEARNING_RATE):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
    
    def forward(self, input_ids, attention_mask,
                decoder_attention_mask=None, labels=None):
        """
        forward method is mainly responsible for performing the forward pass 
        of the model. During the forward pass, the input data is 
        processed through the model's layers, and our output it will generated.

        inputs id: parameter represents the input sequence or tokenized text.

        attention masked: parameter is an optional mask that indicates which 
        positions in the input are valid and which are padding.

        decoder_attention_mask: This parameter is an optional mask mainly 
        used for the decoder part of a model,
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        return outputs.loss
    
    def _step(self, batch):
        source_input_ids, source_attention_mask, target_input_ids, target_attention_mask = batch
        loss = self(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            decoder_attention_mask=target_attention_mask,
            labels=target_input_ids
        )
        return loss 
    # computes the loss based on the model's output and the provided target values.
    """
    Step class we are using for processing a batch of data and computing the loss for a specific model.

    batch: batch paramter contain the input and target data

    Overall, step function, It takes a batch of data, feeds it to a model, 
    and computes the loss based on the model's output and the provided target values.
    """
        
    def training_step(self, batch, batch_idx):
        """
        training:step: work as a part of a training loop to perform a single 
        step of training on a batch of data.

        batch:  Its represents a batch of training data, 
        which mainly  includes both input and target values.

        batch_idx is the index of the current batch within the training loop.

        'train_loss': This is the name or identifier of the training loss metric.

        loss: its computed loss value for the batch data.

        prog_bar=True: This indicates that the training loss should be 
        displayed in the progress bar during training.

        logger=True: This indicates that the training loss should be logged to the logger.

        """
        loss = self._step(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        validation:step: work as a part of a validation loop to evaluate the 
        model's performance on a batch of validation data.

        batch: Its represents a batch of validation data, mainly it containing input and target values.

        batch_idx is the index of the current batch within the validation loop.

        'val_loss': This is the name  of the validation loss metric.

        loss: The computed loss value for the batch.

        prog_bar=True: This indicates that the validation loss should be 
        displayed in the progress bar during validation.

        logger=True: This indicates that the validation loss should be logged to the logger.
        """
        loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def on_train_epoch_end(self):
        """
        This method is likely a callback method that is executed at the end of
        each training epoch. It is used to perform specific actions or 
        log metrics related to the training process.

        callback_metrics: The value of the training loss is obtained from the 
        callback_metrics dictionary of the self.trainer

        The 'train_loss' key is used to access the training loss value for the current epoch.
        """
        self.log('train_loss_epoch', self.trainer.callback_metrics['train_loss'])
    
    def on_validation_epoch_end(self):
        self.log('val_loss_epoch', self.trainer.callback_metrics['val_loss'])
    
    def configure_optimizers(self):
        """
        This scheduler reduces the learning rate when a metric 
        has stopped improving..

        scheduler's mode is set to 'min', indicating that the metric being monitored should be minimized.

        factor: 0.1, means the learning rate will be reduced by a factor of 0.1 when it will get triggered.

        patience parameter specifies the number of epochs with no improvement 
        in the monitored metric before the learning rate is reduced.

        verbose=True parameter indicates that updates about the learning rate 
        changes will be printed during training.


        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'  # Choose the metric to monitor (e.g., validation loss)
            }
        }


