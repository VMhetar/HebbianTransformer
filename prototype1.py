import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')
tokenizer = AutoTokenizer.from_pretrained('mistral/mistral-7b')

class HebbianLayer(nn.Module):
    def __init__(self, in_features, out_features, lr=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.lr = lr

    def forward(self, x):
        out = F.linear(x, self.weight)
        return out

    def hebbian_update(self, pre, post):
        # pre: input activations
        # post: output activations
        delta_w = self.lr * torch.einsum("bi,bj->ij", post, pre)
        self.weight.data += delta_w
