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
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.lr = lr

    def forward(self, x):
        return F.linear(x, self.weight)
    def oja_update(self, pre, post):
        """
        pre: (batch, in_features)
        post: (batch, out_features)
        """
        # Outer product averaged over batch
        hebb_term = torch.einsum("bi,bj->ij", post, pre) / pre.size(0)
        # Oja stabilization term
        y_squared = (post ** 2).mean(dim=0, keepdim=True)  # (1, out_features)

        decay_term = y_squared.t() * self.weight

        delta_w = self.lr * (hebb_term - decay_term)

        self.weight.data += delta_w
