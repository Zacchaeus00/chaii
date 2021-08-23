import numpy as np
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import transformers
from transformers import AutoModelForQuestionAnswering, AutoConfig
from data import ChaiiDataRetriever
from madgrad import MADGRAD
from engine import Engine
from utils import seed_everything, log_scores, log_hyp
import datetime
from pprint import pprint
from torch.optim.swa_utils import AveragedModel, SWALR
seed_everything(42)

hyp = {
    'model_checkpoint': '../../input/deepset-xlm-roberta-large-squad2',
    'train_path': '../../input/chaii-hindi-and-tamil-question-answering/chaii-mlqa-xquad-5folds-count_leq15.csv',
    'stage1_checkpoint': '../model/squad1.2tamil-xrobl-ep1-bs4-ga1-lr5e-06-adamw-wd0.01-cosann-wu0.1-dropoutTrue-evalsteps1000/best_jaccard.pt',
    'max_length': 512,
    'doc_stride': 128,
    'epochs': 2,
    'batch_size': 4,
    'accumulation_steps': 1,
    'lr': 1e-6,
    'optimizer': 'adamw',
    'weight_decay': 0.01,
    'scheduler': 'cosann',
    'warmup_ratio': 0.05,
    'dropout': True,
    'eval_steps': 100
}
experiment_name = 'squad1.2tamil-stage2-xrobl-ep{}-bs{}-ga{}-lr{}-{}-wd{}-{}-wu{}-dropout{}-evalsteps{}'.format(
    hyp['epochs'],
    hyp['batch_size'],
    hyp['accumulation_steps'],
    hyp['lr'],
    hyp['optimizer'],
    hyp['weight_decay'],
    hyp['scheduler'],
    hyp['warmup_ratio'],
    hyp['dropout'],
    hyp['eval_steps'],
)
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
    data_retriever.prepare_data(fold, only_chaii=True, lang='tamil')
    train_dataloader = data_retriever.train_dataloader()
    val_dataloader = data_retriever.val_dataloader()
    predict_dataloader = data_retriever.predict_dataloader()
    cfg = AutoConfig.from_pretrained(hyp['model_checkpoint'])
    if not hyp['dropout']:
        cfg.hidden_dropout_prob = 0
        cfg.attention_probs_dropout_prob = 0
    model = AutoModelForQuestionAnswering.from_pretrained(hyp['model_checkpoint'], config=cfg)
    model.load_state_dict(torch.load(hyp['stage1_checkpoint']))

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
    raw_predictions = engine.predict(predict_dataloader)
    best_score, lang_scores = data_retriever.evaluate_jaccard(raw_predictions)
    print(f'initial score {best_score}')
    for epoch in range(hyp['epochs']):
        best_score = engine.train_evaluate(train_dataloader, 
                                            predict_dataloader, 
                                            data_retriever, 
                                            hyp['eval_steps'], 
                                            best_score, 
                                            out_dir+f'fold{fold}.pt', 
                                            accumulation_steps=hyp['accumulation_steps'])

    print(f'fold {fold} best score {best_score}')
    oof_scores[fold] = best_score
print(f'{folds} fold cv jaccard {oof_scores.mean()}')
log_hyp(out_dir, hyp)
log_scores(out_dir, oof_scores)


    