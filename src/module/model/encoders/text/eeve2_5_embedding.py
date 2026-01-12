import torch
from torch import nn
from transformers import AutoModel

from module.model.encoders.base import BaseEncoder


class EEVEEmbedddingEncoder(BaseEncoder):
    modality = "text"

    def __init__(self, out_features: int = 256) -> None:
        super().__init__()
        self._out_features = out_features
        self.embed_tokens = self._get_embedding()
        self.layers = nn.Linear(self.embed_tokens.embedding_dim, out_features, bias=False)

    def _get_embedding(self):
        model = AutoModel.from_pretrained("yanolja/YanoljaNEXT-EEVE-2.8B")
        return model.embed_tokens

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.embed_tokens(input_ids)
        out = self.layers(out)
        return out
