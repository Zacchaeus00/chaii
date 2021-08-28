import transformers
from transformers import AutoModel, AutoConfig, AutoModelForQuestionAnswering
import torch
import torch.nn as nn
import numpy as np

class Output:
    pass

class ChaiiModel(nn.Module):
    def __init__(self, model_name, config):
        super(ChaiiModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.output = nn.Linear(config.hidden_size, 2)
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        transformer_out = self.transformer(input_ids, attention_mask)
        sequence_output = transformer_out[0]
        logits = self.output(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # total_loss = (start_loss + end_loss) / 2
            total_loss = (start_loss*end_loss) ** 0.5
        else:
            total_loss = None

        output = Output()
        output.loss = total_loss
        output.start_logits = start_logits
        output.end_logits = end_logits
        return output

class ChaiiModelLoadHead(nn.Module):
    def __init__(self, model_name, config):
        super(ChaiiModelLoadHead, self).__init__()
        self.transformer = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        output = self.transformer(input_ids, attention_mask)
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(output.start_logits, start_positions)
            end_loss = loss_fct(output.end_logits, end_positions)
            total_loss = (start_loss*end_loss) ** 0.5
        else:
            total_loss = None
        myoutput = Output()
        myoutput.loss = total_loss
        myoutput.start_logits = output.start_logits
        myoutput.end_logits = output.end_logits
        return myoutput