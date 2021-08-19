import numpy as np
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig
from data import ChaiiDataRetriever
from madgrad import MADGRAD
from engine import Engine
from utils import seed_everything, log_scores, log_hyp
import datetime
from pprint import pprint
seed_everything(42)

hyp = {
    'model_checkpoint': '../../input/deepset-xlm-roberta-large-squad2',
    'train_path': '../../input/chaii-hindi-and-tamil-question-answering/chaii-mlqa-xquad-5folds-count_leq15.csv',
    'experiment_name': 'xrobl-ep3-bs4-lr2.71e-6-adamw-wd1e-2-cosann-wu5e-2-nodropout',
    'max_length': 512,
    'doc_stride': 128,
    'epochs': 3,
    'batch_size': 4,
    'accumulation_steps': 1,
    'lr': 2.71e-6,
    'optimizer': 'adamw',
    'weight_decay': 0.01,
    'scheduler': 'cosann',
    'warmup_ratio': 0.05,
    'dropout': False
}
experiment_name = hyp['experiment_name']
out_dir = f'../model/{experiment_name}/'

print('-'*40)
print(datetime.datetime.now())
pprint(hyp)
print('-'*40)

data_retriever = ChaiiDataRetriever(hyp['model_checkpoint'], hyp['train_path'], hyp['max_length'], hyp['doc_stride'], hyp['batch_size'])
folds = 5
oof_scores = np.zeros(folds)
for fold in range(folds):
    print("fold", fold)
    data_retriever.prepare_data(fold)
    train_dataloader = data_retriever.train_dataloader()
    val_dataloader = data_retriever.val_dataloader()
    predict_dataloader = data_retriever.predict_dataloader()
    cfg = AutoConfig.from_pretrained(hyp['model_checkpoint'])
    if not hyp['dropout']:
        cfg.hidden_dropout_prob = 0
        cfg.attention_probs_dropout_prob = 0
    model = AutoModelForQuestionAnswering.from_pretrained(hyp['model_checkpoint'], config=cfg)

    num_training_steps = hyp['epochs'] * len(train_dataloader)
    num_warmup_steps = int(hyp['warmup_ratio'] * num_training_steps)
    if hyp['optimizer'] == 'madgrad':
        optimizer = MADGRAD(model.parameters(), lr=hyp['lr'], weight_decay=hyp['weight_decay'])
    elif hyp['optimizer'] == 'adamw':
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
    if hyp['scheduler'] == 'cosann':
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    elif hyp['scheduler'] == 'linann':
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    else:
        scheduler = None

    engine = Engine(model, optimizer, scheduler, 'cuda')
    vloss = engine.evaluate(val_dataloader)
    raw_predictions = engine.predict(predict_dataloader)
    score, lang_scores = data_retriever.evaluate_jaccard(raw_predictions)
    print(f'initial vloss {vloss}, score {score}')
    print(lang_scores)
    best_score = 0
    for epoch in range(hyp['epochs']):
        tloss = engine.train(train_dataloader, accumulation_steps=hyp['accumulation_steps'])
        vloss = engine.evaluate(val_dataloader)
        raw_predictions = engine.predict(predict_dataloader)
        score, lang_scores = data_retriever.evaluate_jaccard(raw_predictions)
        print(f'epoch {epoch}, tloss {tloss}, vloss {vloss}, score {score}')
        if score > best_score:
            best_score = score
            engine.save(out_dir+f'fold{fold}.pt')

    print(f'fold {fold} best score {best_score}')
    oof_scores[fold] = best_score
print(f'{folds} fold cv jaccard {oof_scores.mean()}')
log_hyp(out_dir, hyp)
log_scores(out_dir, oof_scores)


    