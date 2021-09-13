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
import logging
seed_everything(42)

hyp = {
    'model_checkpoint': '/gpfsnyu/scratch/yw3642/chaii/input/microsoft-infoxlm-large',
    'train_path': '../../input/chaii-hindi-and-tamil-question-answering/chaii-mlqa-xquad-5folds-count_leq15.csv',
    'experiment_name': 'infoxlm512enta-ep3-bs4-ga1-lr1e-05-adamw-wd0-cosann-wu0.1-dropout0.1-evalsteps1000-metricnonzero_jaccard_per-geolossFalse',
    'max_length': 512,
    'doc_stride': 128,
    'batch_size': 32,
    'swa': False
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
scores = []
all_df = pd.DataFrame()
for fold in range(folds):
    print("fold", fold)
    data_retriever.prepare_data(fold, only_chaii=True)
    predict_dataloader = data_retriever.predict_dataloader()
    model = AutoModelForQuestionAnswering.from_pretrained(hyp['model_checkpoint'])
    if hyp['swa']:
        model = AveragedModel(model)
    model.load_state_dict(torch.load(os.path.join(out_dir, f'fold{fold}.pt')))

    engine = Engine(model, None, None, 'cuda')
    raw_predictions = engine.predict(predict_dataloader)
    score, lang_scores, df = data_retriever.evaluate_jaccard(raw_predictions, return_predictions=True)
    all_df = pd.concat([all_df, df], axis=0)
    hindi_scores.append(lang_scores['hindi'])
    tamil_scores.append(lang_scores['tamil'])
    scores.append(score)
    print(score)
    print(lang_scores)
logging.basicConfig(filename=os.path.join(out_dir, 'evaluate.log'), level=logging.DEBUG)
logging.info('hindi mean: {}'.format(np.mean(hindi_scores)))
logging.info('tamil mean: {}'.format(np.mean(tamil_scores)))
logging.info('macro mean: {}'.format((np.mean(hindi_scores)+np.mean(tamil_scores))/2))
logging.info('micro mean: {}'.format(np.mean(scores)))
all_df.to_csv(os.path.join(out_dir, 'oof_predictions.csv'), index=False)