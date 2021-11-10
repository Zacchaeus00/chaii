import torch
from transformers import AutoModelForQuestionAnswering
import argparse
from glob import glob
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='')
    arg = parser.add_argument
    arg('-m', '--model_checkpoint', required=True, type=str)
    arg('-w', '--weight_path', required=True, type=str)
    return parser.parse_args()
args = parse_args()

model = AutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint, torchscript=True)
input_ids = torch.tensor([[i for i in range(512)] for _ in range(16)])
attention_mask = torch.tensor([[1 for i in range(512)] for _ in range(16)])
dummy_input = [input_ids, attention_mask]
weight_paths = glob(os.path.join(args.weight_path, '*.pt'))
prename = [x for x in args.weight_path.split('/') if x != ''][-1]
outdir = f"../model/torchscript/{prename}/"
Path(outdir).mkdir(parents=True, exist_ok=True)
map_loc = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


for i, wp in enumerate(weight_paths):
    print(f'converting model {i}')
    name = wp.split('/')[-1].split('.')[0]
    model.load_state_dict(torch.load(wp, map_location=map_loc))
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, os.path.join(outdir, f'{name}.pt'))
