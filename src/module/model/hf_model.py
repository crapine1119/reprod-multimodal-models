from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import ModelOutput

from module.model.hf_config import MultimodalPreTrainedConfig
from module.utils.import_util import instantiate


class MultimodalPreTrainedModel(PreTrainedModel):
    """
    멀티모달 분류용 HF 모델.
    - _target_ 기반 스펙으로 encoders/fusion/heads를 조립
    - save_pretrained()/from_pretrained() 재현 가능
    """

    config_class = MultimodalPreTrainedConfig
    base_model_prefix = "multimodal_interview"

    def __init__(self, config: MultimodalPreTrainedConfig) -> None:
        super().__init__(config)

        self.modalities = list(config.modalities)
        allowed = config.allowed_target_prefixes

        # 1) encoders
        encoders: Dict[str, nn.Module] = {}
        for m in self.modalities:
            if m not in config.encoders:
                raise ValueError(f"encoder_specs missing modality '{m}'. got keys={sorted(config.encoders.keys())}")

            encoder_func = config.encoders[m]
            encoders[m] = instantiate(encoder_func, allowed_prefixes=allowed)

            if not isinstance(encoders[m], nn.Module):
                raise TypeError(f"encoder for '{m}' must be nn.Module. got {type(encoders[m])}")
        self.encoders = nn.ModuleDict(encoders)

        # 2) fusion
        if not config.fusion:
            raise ValueError("fusion_spec is required")
        self.fusion = instantiate(config.fusion, allowed_prefixes=allowed)
        if not isinstance(self.fusion, nn.Module):
            raise TypeError(f"fusion must be nn.Module. got {type(self.fusion)}")

        # 3) heads
        if not config.head:
            raise ValueError("heads_spec is required")

        self.head = instantiate(config.head, allowed_prefixes=allowed)
        if not isinstance(self.head, nn.Module):
            raise TypeError(f"heads must be nn.Module. got {type(self.head)}")

        self.post_init()

    def _forward_multimodal_dict(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        embed_dict: Dict[str, torch.Tensor] = {}
        input_keys = set(x.keys())

        for modality_name, encoder in self.encoders.items():
            if modality_name not in input_keys:
                raise ValueError(f"input missing modality '{modality_name}'. got keys={sorted(input_keys)}")

            modality_input = x[modality_name]

            # 인코더가 (x) 또는 (x, mask) 모두 대응하도록 처리
            embed_dict[modality_name] = encoder(modality_input)
            embed_dict[f"{modality_name}_mask"] = x.get(f"{modality_name}_mask", None)

        fused = self.fusion(embed_dict)
        logits = self.head(fused)
        return logits

    def forward(
        self,
        # text
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # vision
        pixel_values_videos: Optional[torch.Tensor] = None,
        # audio
        input_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        # labels
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput | Tuple[torch.Tensor, ...]:
        if return_dict is None:
            return_dict = self.config.use_return_dict

        # HF 입력 → 내부 모달 딕셔너리
        x: Dict[str, torch.Tensor] = {}

        if "text" in self.modalities:
            if input_ids is None:
                raise ValueError("input_ids is required because 'text' modality is enabled.")
            x["text"] = input_ids
            if attention_mask is not None:
                x["text_mask"] = attention_mask

        if "vision" in self.modalities:
            if pixel_values_videos is None:
                raise ValueError("pixel_values is required because 'vision' modality is enabled.")
            x["vision"] = pixel_values_videos

        if "audio" in self.modalities:
            if input_features is None:
                raise ValueError("input_values is required because 'audio' modality is enabled.")
            x["audio"] = input_features
            if audio_attention_mask is not None:
                x["audio_mask"] = audio_attention_mask

        logits = self._forward_multimodal_dict(x)

        loss: Optional[torch.Tensor] = None
        if not return_dict:
            if loss is None:
                return (logits,)
            return (loss, logits)

        return ModelOutput(loss=loss, logits=logits)
