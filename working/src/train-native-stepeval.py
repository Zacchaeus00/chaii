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
from mymodel import ChaiiModel, ChaiiModelLoadHead
seed_everything(42)

hyp = {
    # 'model_checkpoint': '../../input/microsoft-infoxlm-large-squad2-512',
    'model_checkpoint': '../../input/google-rembert-squad2-512',
    'train_path': '../../input/chaii-hindi-and-tamil-question-answering/chaii-mlqa-xquad-5folds-count_leq15.csv',
    'max_length': 512,
    'doc_stride': 128,
    'epochs': 3,
    'batch_size': 4,
    'accumulation_steps': 1,
    'lr': 1e-5,
    'optimizer': 'adamw',
    'weight_decay': 0,
    'scheduler': 'cosann',
    'warmup_ratio': 0.1,
    'dropout': 0.1,
    'eval_steps': 1000,
    'metric': 'nonzero_jaccard_per',
    'geoloss': True
}
experiment_name = 'rembert512-ep{}-bs{}-ga{}-lr{}-{}-wd{}-{}-wu{}-dropout{}-evalsteps{}-metric{}-geoloss{}'.format(
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
    hyp["metric"],
    hyp['geoloss']
)
out_dir = f'../model/{experiment_name}/'

print('-'*40)
print(datetime.datetime.now())
print(experiment_name)
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
    cfg.hidden_dropout_prob = hyp['dropout']
    cfg.attention_probs_dropout_prob = hyp['dropout']
    if hyp['geoloss']:
        model = ChaiiModelLoadHead(model_name=hyp['model_checkpoint'], config=cfg)
    else:
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
    elif hyp['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr'])
    if hyp['scheduler'] == 'cosann':
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    elif hyp['scheduler'] == 'linann':
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    else:
        scheduler = None

    engine = Engine(model, optimizer, scheduler, 'cuda')
    raw_predictions = engine.predict(predict_dataloader)
    best_score, lang_scores, df = data_retriever.evaluate_jaccard(raw_predictions, return_predictions=True)
    nonzero_jaccard_per = len(df[df['jaccard']!=0]) / len(df)
    print(f'initial mean jaccard {best_score}')
    print(f'initial nonzero jaccard percentage {nonzero_jaccard_per}')
    best_metric = best_score if hyp["metric"] == 'mean_jaccard' else nonzero_jaccard_per
    print(f'using metric: {hyp["metric"]}')
    for epoch in range(hyp['epochs']):
        best_metric = engine.train_evaluate(train_dataloader, 
                                            predict_dataloader, 
                                            data_retriever, 
                                            hyp['eval_steps'], 
                                            best_metric, 
                                            out_dir+f'fold{fold}.pt', 
                                            hyp["metric"],
                                            accumulation_steps=hyp['accumulation_steps'])

    print(f'fold {fold} best {hyp["metric"]} {best_metric}')
    oof_scores[fold] = best_metric
    # torch.save(model.state_dict(), out_dir+f'fold{fold}_last.pt')
print(f'{folds} fold cv {hyp["metric"]}: {oof_scores.mean()}')
log_hyp(out_dir, hyp)
log_scores(out_dir, oof_scores)


    