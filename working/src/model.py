import transformers
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
import numpy as np


class ChaiiModel(nn.Module):
    def __init__(self, model_name, hidden_dropout_prob, layer_norm_eps):
        super(ChaiiModel, self).__init__()

        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )
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
            total_loss = (start_loss + end_loss) / 2
        else:
            total_loss = None

        return {
            'loss': total_loss,
            'start_logits': start_logits,
            'end_logits': end_logits
        }
