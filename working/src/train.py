import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
from datasets import Dataset
import collections
from tqdm.auto import tqdm
import os
from utils import jaccard, convert_answers, prepare_train_features, prepare_validation_features, postprocess_qa_predictions, log_score
model_checkpoint = 'deepset/xlm-roberta-base-squad2'
out_dir = os.path.join('../model', '-'.join(model_checkpoint.split('/')))
folds = 10
args = TrainingArguments(output_dir=out_dir,
                        learning_rate=1e-5,
                        warmup_ratio=0.1,
                        gradient_accumulation_steps=1,
                        per_device_train_batch_size=32,
                        per_device_eval_batch_size=32,
                        num_train_epochs=1,
                        weight_decay=0.01,
                        fp16=True,
                        report_to='tensorboard',
                        load_best_model_at_end=True,
                        evaluation_strategy="steps",
                        eval_steps=100,
                        dataloader_num_workers=8,
                        save_total_limit=1,
                        save_strategy='steps',
                        save_steps=100
                        )
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer.padding_side == "right"

train = pd.read_csv('../../input/chaii-hindi-and-tamil-question-answering/train_folds.csv')
train['answers'] = train[['answer_start', 'answer_text']].apply(convert_answers, axis=1)

oof_scores = np.zeros(folds)
for fold in range(folds):
    print(f'running fold {fold}')
    df_train = train[train['kfold']!=fold].reset_index(drop=True)
    df_valid = train[train['kfold']==fold].reset_index(drop=True)
    print(f'train/val samples: {len(df_train)}/{len(df_valid)}')
    train_dataset = Dataset.from_pandas(df_train)
    valid_dataset = Dataset.from_pandas(df_valid)
    tokenized_train_ds = train_dataset.map(lambda x: prepare_train_features(x, tokenizer, max_length, doc_stride, pad_on_right), 
                                           batched=True, 
                                           remove_columns=train_dataset.column_names)
    tokenized_valid_ds = valid_dataset.map(lambda x: prepare_train_features(x, tokenizer, max_length, doc_stride, pad_on_right), 
                                           batched=True, 
                                           remove_columns=train_dataset.column_names)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    trainer = Trainer(model,
                      args,
                      train_dataset=tokenized_train_ds,
                      eval_dataset=tokenized_valid_ds,
                      data_collator=default_data_collator,
                      tokenizer=tokenizer,
                      )
    trainer.train()
    # trainer.save_model(os.path.join(out_dir, f'fold{fold}'))
    
    validation_features = valid_dataset.map(lambda x: prepare_validation_features(x, tokenizer, max_length, doc_stride, pad_on_right),
                                            batched=True,
                                            remove_columns=valid_dataset.column_names
                                            )
    valid_feats_small = validation_features.map(lambda example: example, remove_columns=['example_id', 'offset_mapping'])
    raw_predictions = trainer.predict(valid_feats_small)
    example_id_to_index = {k: i for i, k in enumerate(valid_dataset["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(validation_features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    final_predictions = postprocess_qa_predictions(valid_dataset, 
                                                   validation_features, 
                                                   raw_predictions.predictions,
                                                   features_per_example,
                                                   tokenizer,
                                                   n_best_size=20,
                                                   max_answer_length=30)
    references = [{"id": ex["id"], "answer": ex["answers"]['text'][0]} for ex in valid_dataset]
    res = pd.DataFrame(references)
    res['prediction'] = res['id'].apply(lambda r: final_predictions[r])
    res['jaccard'] = res[['answer', 'prediction']].apply(jaccard, axis=1)
    oof_scores[fold] = res.jaccard.mean()
    print(f'jaccard: {res.jaccard.mean()}')
print(f'cv jaccard: {oof_scores.mean()}')
log_score(out_dir, oof_scores.mean())
