import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from utils import jaccard
from data import ChaiiDataRetriever
model_name = '../input'


data_retriever = ChaiiDataRetriever(model_name, train_path, max_length, doc_stride, batch_size)