import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os
import torch

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('-m', '--model_checkpoint', required=True, type=str)
    arg('-w', '--weight_path', required=True, type=str)
    return parser.parse_args()
args = parse_args()

model = AutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint)

model.load_state_dict(torch.load(os.path.join(args.weight_path, '0.pt'), map_location='cpu'))

tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

tokenizer.push_to_hub("muril-large-cased-hita-qa")
model.push_to_hub("muril-large-cased-hita-qa")