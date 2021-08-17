import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import Dataset
from utils import prepare_train_features, prepare_validation_features, convert_answers

class ChaiiDataRetriever:
    def __init__(self, model_name, train_path, max_length, doc_stride, batch_size):
        self.model_name = model_name
        self.train = pd.read_csv(train_path)
        self.train['answers'] = self.train[['answer_start', 'answer_text']].apply(convert_answers, axis=1)
        self.train['id'] = self.train.index
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pad_on_right = self.tokenizer.padding_side == "right"
        
    def prepare_data(self, fold):
        df_train = self.train[self.train['fold']!=fold].reset_index(drop=True)
        df_valid = self.train[self.train['fold']==fold].reset_index(drop=True)
        self.train_dataset = Dataset.from_pandas(df_train)
        self.valid_dataset = Dataset.from_pandas(df_valid)
        self.tokenized_train_ds = self.train_dataset.map(lambda x: prepare_train_features(x, self.tokenizer, self.max_length, self.doc_stride, self.pad_on_right),
                                                        batched=True, 
                                                        remove_columns=self.train_dataset.column_names)
        self.tokenized_valid_ds = self.valid_dataset.map(lambda x: prepare_train_features(x, self.tokenizer, self.max_length, self.doc_stride, self.pad_on_right), 
                                                        batched=True, 
                                                        remove_columns=self.train_dataset.column_names)
        self.validation_features = self.valid_dataset.map(lambda x: prepare_validation_features(x, self.tokenizer, self.max_length, self.doc_stride, self.pad_on_right),
                                                        batched=True,
                                                        remove_columns=self.valid_dataset.column_names
                                                        )
        self.valid_feats_small = self.validation_features.map(lambda example: example, remove_columns=['example_id', 'offset_mapping'])
        self.tokenized_train_ds.set_format(type='torch')
        self.tokenized_valid_ds.set_format(type='torch')
        self.valid_feats_small.set_format(type='torch')
    def train_dataloader(self):
        return DataLoader(self.tokenized_train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.tokenized_valid_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.valid_feats_small, batch_size=self.batch_size)