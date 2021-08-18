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
import optuna
seed_everything(42)
model_checkpoint = '../../input/deepset-xlm-roberta-large-squad2'
train_path = '../../input/chaii-hindi-and-tamil-question-answering/chaii-mlqa-xquad-5folds-count_leq15.csv'
experiment_name = 'xrob-large-optuna-subspace-adamw-cosann0.05wu-5ep-fold0'
out_dir = f'../model/{experiment_name}/'

max_length = 512
doc_stride = 128
batch_size = 4

print('-'*40)
print(datetime.datetime.now())
print(experiment_name)
print('-'*40)

data_retriever = ChaiiDataRetriever(model_checkpoint, train_path, max_length, doc_stride, batch_size)
data_retriever.prepare_data(0)
train_dataloader = data_retriever.train_dataloader()
val_dataloader = data_retriever.val_dataloader()
predict_dataloader = data_retriever.predict_dataloader()
folds = 1
epochs = 5

def objective(trial):
    print(datetime.datetime.now())
    hyp = {
        'lr': trial.suggest_loguniform('lr', 1e-6, 1e-4),
        'accumulation_steps': trial.suggest_categorical('accumulation_steps', [1, 2, 4, 6, 8]),
        # 'optimizer': trial.suggest_categorical('optimizer', ['adamw', 'madgrad']),
        'optimizer': 'adamw',
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-1),
        # 'weight_decay': 0.0,
        # 'scheduler': trial.suggest_categorical('scheduler', ['cosann', 'linann']),
        'scheduler': 'cosann',
        # 'warmup_ratio': trial.suggest_uniform('warmup_ratio', 0, 0.5),
        'warmup_ratio': 0.05,
        # 'grad_clip': trial.suggest_uniform('grad_clip', 0.5, 5),
        'grad_clip': 1,
    }
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    num_training_steps = epochs * len(train_dataloader)
    num_warmup_steps = int(hyp['warmup_ratio'] * num_training_steps)
    if hyp['optimizer'] == 'adamw':
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": hyp['weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=hyp['lr'])
    elif hyp['optimizer'] == 'madgrad':
        optimizer = MADGRAD(model.parameters(), lr=hyp['lr'], weight_decay=hyp['weight_decay'])
    
    if hyp['scheduler'] == 'cosann':
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    elif hyp['scheduler'] == 'linann':
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
        
    engine = Engine(model, optimizer, scheduler, 'cuda')

    # vloss = engine.evaluate(val_dataloader)
    # raw_predictions = engine.predict(predict_dataloader)
    # init_score, _ = data_retriever.evaluate_jaccard(raw_predictions)
    # print(f'initial vloss {vloss}, score {init_score}')
    print('initial vloss 1.513, score 0.632')
    init_score = 0.632
    best_score = 0
    for epoch in range(epochs):
        tloss = engine.train(train_dataloader, accumulation_steps=hyp['accumulation_steps'], grad_clip=hyp['grad_clip'])
        vloss = engine.evaluate(val_dataloader)
        raw_predictions = engine.predict(predict_dataloader)
        score, lang_scores = data_retriever.evaluate_jaccard(raw_predictions)
        if score < init_score:
            raise optuna.exceptions.TrialPruned()
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        print(f'epoch {epoch}, tloss {tloss}, vloss {vloss}, score {score}')
        if score > best_score:
            best_score = score
    return best_score

study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), direction='maximize')
study.optimize(objective, n_trials=1000)



    