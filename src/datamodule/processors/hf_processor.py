from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional, Tuple, Union

import torch
from transformers import ProcessorMixin


class BaseMultimodalProcessor(ProcessorMixin, ABC):

    @abstractmethod
    def collate_fn(self, batch) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save_pretrained(self, save_directory: str, **kwargs: Any) -> Tuple[str]:
        raise NotImplementedError


class SimpleMultimodalProcessor(BaseMultimodalProcessor):
    """
    simple implementation using hugingface style processors.
    Refer to Qwen2.5 Omni
    """

    attributes = ["video_processor", "feature_extractor", "tokenizer"]
    video_processor_class = "AutoVideoProcessor"
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        video_processor: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        max_text_length: Optional[int] = 128,
        chat_template: Optional[str] = None,
    ) -> None:
        super().__init__(video_processor, feature_extractor, tokenizer, chat_template=chat_template)
        self._max_text_length = max_text_length

    def __call__(
        self,
        videos: Optional[Union[torch.Tensor | list[torch.Tensor]]] = None,
        audio: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:

        videos_inputs = {}
        if videos is not None:
            videos_inputs = self.video_processor(videos)

        audio_inputs = {}
        if audio is not None:
            audio_inputs = self.feature_extractor(
                audio, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
            )

        text_inputs = {}
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                padding="max_length",
                padding_side="right",
                truncation=True,
                max_length=self._max_text_length,
                return_tensors="pt",
            )
            text_inputs["text"] = text

        return {**videos_inputs, **audio_inputs, **text_inputs}

    # TODO
    def save_pretrained(self, save_directory: str, **kwargs: Any) -> Tuple[str]:
        pass

    def collate_fn(self, batch: list) -> dict[str, Any]:
        collated = defaultdict(list)

        for b in batch:
            processed = self(videos=b["videos"], audio=b["audio"], text=b["text"])
            for k, v in processed.items():
                if isinstance(v, torch.Tensor):
                    v = v.squeeze()

                collated[k].append(v)
            collated["labels"].append(b["labels"])

        list_columns = ["video_grid_thw", "text"]
        long_columns = ["labels"]

        out = {}
        for k, v in collated.items():
            if k in list_columns:
                out[k] = v
            elif k in long_columns:
                out[k] = torch.tensor(v, dtype=torch.long)
            else:
                out[k] = torch.stack(v, dim=0)
        # Audio to BTC
        out["input_features"] = out["input_features"].permute(0, 2, 1)
        return out


if __name__ == "__main__":
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
    from transformers import WhisperFeatureExtractor, AutoTokenizer
    from datamodule.datasets.aihub_multimodal import AIHubMultimodalDataset
    from torch.utils.data import DataLoader

    ds = AIHubMultimodalDataset(
        data_dir="../../../assets/aihub/2019-01-005.멀티모달영상_sample/원천데이터",
        label_dir="../../../assets/aihub/2019-01-005.멀티모달영상_sample/라벨데이터",
        data_extension="mp4",
        label_extension="json",
        label_prefix="clip_",
        end_video_sec=64.0,
        video_chunk_sec=2,
        fps=4,
        video_resize_rate=0.1,
        sampling_rate=16000,
        require_label=False,
    )
    data = ds[7]

    processor = SimpleMultimodalProcessor(
        video_processor=Qwen2VLVideoProcessor(merge_size=4),
        feature_extractor=WhisperFeatureExtractor(chunk_length=4, sampling_rate=16000),
        tokenizer=AutoTokenizer.from_pretrained("yanolja/YanoljaNEXT-EEVE-2.8B"),
        max_text_length=16,
    )

    # output = processor(**data)
    # print(output)

    dataloader = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        collate_fn=processor.collate_fn,
    )

    print(next(iter(dataloader)))
