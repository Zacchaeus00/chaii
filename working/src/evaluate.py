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
    'experiment_name': 'xrobl-ep3-bs4-lr2.71e-6-adamw-wd1e-2-cosann-wu5e-2-stage2-ep1-lr1e-6',
    'max_length': 512,
    'doc_stride': 128,
    'batch_size': 32,
}
experiment_name = hyp['experiment_name']
out_dir = f'../model/{experiment_name}/'

print('-'*40)
print(datetime.datetime.now())
pprint(hyp)
print('-'*40)

data_retriever = ChaiiDataRetriever(hyp['model_checkpoint'], hyp['train_path'], hyp['max_length'], hyp['doc_stride'], hyp['batch_size'])
folds = 5
hindi_scores = []
tamil_scores = []
for fold in range(folds):
    print("fold", fold)
    data_retriever.prepare_data(fold, only_chaii=True)
    predict_dataloader = data_retriever.predict_dataloader()
    model = AutoModelForQuestionAnswering.from_pretrained(hyp['model_checkpoint'])
    model.load_state_dict(torch.load(os.path.join(out_dir, f'fold{fold}.pt')))

    engine = Engine(model, None, None, 'cuda')
    raw_predictions = engine.predict(predict_dataloader)
    score, lang_scores = data_retriever.evaluate_jaccard(raw_predictions)
    hindi_scores.append(lang_scores['hindi'])
    tamil_scores.append(lang_scores['tamil'])
    print(score)
    print(lang_scores)
print('hindi:', np.mean(hindi_scores))
print('tamil:', np.mean(tamil_scores))
