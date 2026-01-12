import torch
from torch import nn

from module.model.encoders.base import ModalityName


class SimpleTimeConcat(nn.Module):
    def __init__(self, sequence_length: int, out_features: int):
        super().__init__()
        self._sequence_length = sequence_length
        self.pool = nn.AdaptiveMaxPool1d(sequence_length)
        self.linear1 = nn.Linear(in_features=sequence_length * out_features, out_features=out_features)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(in_features=out_features, out_features=out_features)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        # each output should be B, T, C (T dim to be concatenated)
        out = torch.concat(
            [x[modality_name] for modality_name in ModalityName.__args__ if modality_name in x.keys()], dim=1
        )
        b, *_ = out.size()
        out = self.pool(out.permute(0, 2, 1))
        out = out.reshape(b, -1)
        out = self.act(self.linear1(out))
        out = out + self.linear2(out)
        return out
