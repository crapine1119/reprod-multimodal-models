from typing import ClassVar, Literal, Optional

import torch
from torch import nn

ModalityName = Literal["vision", "audio", "text"]


class BaseEncoder(nn.Module):
    # 자동 등록에 필요한 최소 메타
    modality: ClassVar[ModalityName]  # 각 인코더가 어떤 모달리티인지
    # 기본 키는 클래스명. 필요하면 명시적으로 고정 키를 줄 수도 있음
    component_id: ClassVar[Optional[str]] = None


class DummyEncoder(BaseEncoder):
    modality = "text"

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn((1, self.out_features))
