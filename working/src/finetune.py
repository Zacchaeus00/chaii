import json
from pathlib import Path
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig, TrainingArguments, Trainer, default_data_collator
from transformers import XLMRobertaTokenizerFast, XLMRobertaForQuestionAnswering
import os
from utils import seed_everything
from pprint import pprint
import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed_everything(42)

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    imp = 0
    p = 0
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                is_impossible = qa['is_impossible']
                question = qa['question']
                contexts.append(context)
                questions.append(question)
                if is_impossible:
                    imp += 1        
                    answers.append({'text': [], 'answer_start': []})
                else:
                    p += 1
                    text = qa['answers'][0]['text']
                    answer_start = qa['answers'][0]['answer_start']
                    answers.append({'text': [text], 'answer_start': [answer_start]})
    print(f'imp {imp}, p {p}')
    return {'id': [i for i in range(len(contexts))], 'context': contexts, 'question': questions, 'answers': answers}

def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding = False
        # padding="max_length",
        #  padding="longest",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


if __name__ == '__main__':
    train = read_squad('../../input/squad2/train-v2.0.json')
    train_dataset = Dataset.from_dict(train)
    # model_checkpoint = '../../input/google-rembert'
    # model_checkpoint = '../../input/microsoft-infoxlm-large'
    # model_checkpoint = '../../input/xlm-roberta-large'
    model_checkpoint = '../../input/google-muril-base-case'
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_checkpoint) if 'info' in model_checkpoint else AutoTokenizer.from_pretrained(model_checkpoint)
    config = AutoConfig.from_pretrained(model_checkpoint)
    pad_on_right = tokenizer.padding_side == "right"
    # max_length = config.max_position_embeddings 
    max_length = 512
    print('max length:', max_length)
    doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
    tokenized_train_ds = train_dataset.map(prepare_train_features, batched=True, remove_columns=train_dataset.column_names)
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint, config=config)

    experiment_name = 'squad2-512'
    hyp = dict(
        output_dir = f"{model_checkpoint}-{experiment_name}",
        evaluation_strategy = "no",
        save_strategy = "epoch",
        learning_rate=1e-5,
        warmup_ratio=0.2,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        report_to='none',
        dataloader_num_workers=8
    )
    print('-'*40)
    print(datetime.datetime.now())
    pprint(hyp)
    print('-'*40)
    args = TrainingArguments(
        **hyp
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train_ds,
        # eval_dataset=tokenized_valid_ds,
        # data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()