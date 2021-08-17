import numpy as np
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from data import ChaiiDataRetriever
from madgrad import MADGRAD
from engine import Engine
from utils import seed_everything, log_scores
import datetime
seed_everything(42)
model_checkpoint = '../../input/deepset-xlm-roberta-base-squad2'
train_path = '../../input/chaii-hindi-and-tamil-question-answering/chaii-mlqa-xquad-5folds.csv'
experiment_name = 'xrob-base-bs24-lr5e-6-madgrad'
out_dir = f'../model/{experiment_name}/'

max_length = 512
doc_stride = 128
batch_size = 24
lr = 5e-6

print('-'*40)
print(datetime.datetime.now())
print(experiment_name)
print(f'lr {lr}, bs {batch_size}')
print('-'*40)

data_retriever = ChaiiDataRetriever(model_checkpoint, train_path, max_length, doc_stride, batch_size)
folds = 5
epochs = 10
oof_scores = np.zeros(folds)
for fold in range(folds):
    print("fold", fold)
    data_retriever.prepare_data(fold)
    train_dataloader = data_retriever.train_dataloader()
    val_dataloader = data_retriever.val_dataloader()
    predict_dataloader = data_retriever.predict_dataloader()
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    num_training_steps = epochs * len(train_dataloader)
    # optimizer = MADGRAD(model.parameters(), lr)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)
    # scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=0.1*num_training_steps, num_training_steps=num_training_steps, num_cycles=3)
    scheduler = None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    engine = Engine(model, optimizer, scheduler, device)

    vloss = engine.evaluate(val_dataloader)
    raw_predictions = engine.predict(predict_dataloader)
    score, lang_scores = data_retriever.evaluate_jaccard(raw_predictions)
    print(f'initial vloss {vloss}, score {score}')
    print(lang_scores)
    best_score = 0
    es_counter = 0
    es_patience = 3
    for epoch in range(epochs):
        es_counter += 1
        tloss = engine.train(train_dataloader)
        vloss = engine.evaluate(val_dataloader)
        raw_predictions = engine.predict(predict_dataloader)
        score, lang_scores = data_retriever.evaluate_jaccard(raw_predictions)
        print(f'epoch {epoch}, tloss {tloss}, vloss {vloss}, score {score}')
        print(lang_scores)
        if score > best_score:
            best_score = score
            engine.save(out_dir+f'fold{fold}.pt')
            es_counter = 0
        else:
            if es_counter == es_patience:
                print('early stopping.')
                break
    print(f'fold {fold} best score {best_score}')
    oof_scores[fold] = best_score
print(f'{folds}-fold cv jaccard {oof_scores.mean()}')
log_scores(out_dir, oof_scores)


    