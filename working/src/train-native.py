import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import jaccard
from data import ChaiiDataRetriever
from madgrad import MADGRAD
from engine import Engine
model_name = '../../input/deepset-xlm-roberta-base-squad2'
train_path = '../../input/chaii-hindi-and-tamil-question-answering/chaii-mlqa-xquad-5folds.csv'
max_length = 512
doc_stride = 128
batch_size = 32

data_retriever = ChaiiDataRetriever(model_name, train_path, max_length, doc_stride, batch_size)
folds = 1
epochs = 5
oof_scores = np.zeros(folds)
for fold in range(folds):
    print("fold", fold)
    data_retriever.prepare_data(fold)
    train_dataloader = data_retriever.train_dataloader()
    val_dataloader = data_retriever.val_dataloader()
    predict_dataloader = data_retriever.predict_dataloader()
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    optimizer = MADGRAD(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_dataloader))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    engine = Engine(model, optimizer, scheduler, device)
    for epoch in range(epochs):
        tloss = engine.train(train_dataloader)
        vloss = engine.evaluate(val_dataloader)
        print(f'epoch {epoch}, tloss {tloss}, vloss {vloss}')

    