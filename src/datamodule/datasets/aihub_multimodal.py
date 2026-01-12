import json
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder, VideoDecoder

LabelIndexMap: dict[str, int] = {
    "null": 0,
    "neutral": 1,
    "happy": 2,
    "angry": 3,
    "dislike": 4,
    "sad": 5,
    "surprise": 6,
    "contempt": 7,
}


@dataclass(frozen=True)
class ChunkSpec:
    chunk_idx: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class SampleIndex:
    chunk: ChunkSpec
    video_path: str
    label_path: Optional[str]
    key: str  # video key (보통 stem)


class FileManangementMixin:
    def _glob_files(self, dir_path: str, extension: str) -> List[str]:
        extension = extension.lstrip(".")
        patterns = [
            os.path.join(dir_path, f"*.{extension}"),
            os.path.join(dir_path, f"*.{extension.lower()}"),
            os.path.join(dir_path, f"*.{extension.upper()}"),
        ]
        out: List[str] = []
        for p in patterns:
            out.extend(glob(p))
        out = sorted(set(out))
        if not out:
            raise FileNotFoundError(f"No files found: dir={dir_path}, ext={extension}")
        return out

    def _build_chunks(self, start_video_sec: float, end_video_sec: float, video_chunk_sec: int) -> List[ChunkSpec]:
        if video_chunk_sec <= 0:
            raise ValueError("sec must be > 0")
        return [
            ChunkSpec(chunk_idx=enum, start_sec=i, end_sec=i + video_chunk_sec)
            for enum, i in enumerate(np.arange(start_video_sec, end_video_sec, video_chunk_sec))
        ]

    def _get_label_index(self, label_stem: str, *, prefix: str = "clip_") -> str:
        """
        예: clip_12.json -> "0012"
        매칭 규칙은 데이터셋마다 다르므로, 필요하면 사용자 쪽에서 교체하십시오.
        """
        if label_stem.startswith(prefix):
            rest = label_stem[len(prefix) :]
            try:
                return f"{int(rest):04d}"
            except ValueError:
                return label_stem
        return label_stem

    def _build_label_map(self, label_paths: list[str], label_prefix: str) -> dict[str, str]:
        """
        video_key -> label_path 매핑.
        """
        label_map: dict[str, str] = {}
        for lp in label_paths:
            stem = Path(lp).stem
            label_index = self._get_label_index(stem, prefix=label_prefix)
            label_map[label_index] = lp
        return label_map


class AIHubMultimodalDataset(Dataset, FileManangementMixin):
    """
    Hugging Face 스타일에 맞추기 위한 Dataset 설계:
    - Dataset은 raw를 반환
      {"video": Tensor(T,H,W,C), "audio": Tensor(L,), "text": str|None, "label": Any|None, "meta": {...}}
    - processor는 Dataset에 넣지 않고 collator에서 배치 단위로 적용

    오디오:
    - torchvision.read_video의 raw_audio 대신 torchcodec AudioDecoder를 사용해 sample_rate로 통일
    - hf:// 같은 원격 경로로 확장할 때도 file-like로 처리 가능
    """

    def __init__(
        self,
        data_dir: str,
        data_extension: str,
        label_dir: str,
        label_extension: str,
        *,
        # 라벨 로딩
        label_prefix: str = "clip_",
        # 라벨 없을 때 처리
        require_label: bool = False,
        # 청크: 사용자가 예시로 0~5초를 2등분했으므로 기본값을 그렇게 둡니다.
        start_video_sec: float = 0.0,
        end_video_sec: float = 60.0,
        video_chunk_sec: int = 4,
        video_resize_rate: float = 0.5,
        fps: float = 8,
    ) -> None:
        super().__init__()
        # files
        self._data_dir = data_dir
        self._data_extension = data_extension
        self._label_dir = label_dir
        self._label_extension = label_extension
        self._label_prefix = label_prefix
        self._require_label = bool(require_label)
        # video
        self._start_video_sec = start_video_sec
        self._end_video_sec = end_video_sec
        self._video_chunk_sec = video_chunk_sec
        self._video_resize_rate = video_resize_rate
        self._fps = fps
        self._target_frames = int(self._video_chunk_sec * self._fps)

        # data
        self._video_paths = self._glob_files(self._data_dir, self._data_extension)
        self._label_paths = self._glob_files(self._label_dir, self._label_extension)
        self._data = self._build_index()
        self._video_fps_cache: float = None

    def __len__(self) -> int:
        return len(self._data)

    def _build_index(self) -> List[SampleIndex]:
        """
        (video, label, chunk) 단위로 샘플 인덱스를 만듭니다.
        """
        chunks = self._build_chunks(
            start_video_sec=float(self._start_video_sec),
            end_video_sec=float(self._end_video_sec),
            video_chunk_sec=int(self._video_chunk_sec),
        )
        label_map = self._build_label_map(self._label_paths, self._label_prefix)

        idx: list[SampleIndex] = []
        for vp in self._video_paths:
            key = Path(vp).stem
            label_path = label_map.get(key)
            if self._require_label and label_path is None:
                continue
            for ch in chunks:
                idx.append(SampleIndex(chunk=ch, video_path=vp, label_path=label_path, key=key))
        return idx

    # def _load_video(self, video_path: str, *, start_sec: float, end_sec: float) -> tuple[torch.Tensor, dict[str, Any]]:
    #     v, _a, info = read_video(
    #         video_path, output_format="TCHW", pts_unit="sec", start_pts=float(start_sec), end_pts=float(end_sec)
    #     )
    #     # resize
    #     t, _, h, w = v.size()
    #     h_pad, w_pad = int(h * self._video_resize_rate), int(w * self._video_resize_rate)
    #     v = Resize(size=(h_pad, w_pad))(v)
    #
    #     # temporal sampling
    #     if t < self._target_frames:
    #         v_pad = torch.zeros(size=(self._target_frames, 3, h_pad, w_pad), dtype=v.dtype)
    #         v_pad[:t] = v
    #         return v_pad, info
    #     else:
    #         temporal_indices = torch.arange(0, t, t / self._target_frames).int()
    #         v_sampled = [v[i] for i in temporal_indices]
    #         return v_sampled, info

    def _load_video(self, video_path: str, *, start_sec: float, end_sec: float) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Returns:
            v: (T, C, H, W) float32 tensor (0~1 범위로 정규화)
            info: 메타데이터 dict
        """
        # TorchCodec 권장: VideoDecoder 사용 :contentReference[oaicite:1]{index=1}
        decoder = VideoDecoder(
            video_path,
            dimension_order="NCHW",  # (N, C, H, W) :contentReference[oaicite:2]{index=2}
            device="cpu",
            num_ffmpeg_threads=1,
            seek_mode="exact",
        )
        md = decoder.metadata

        # 시간 범위 보정
        start = float(max(0.0, start_sec))
        stop = float(end_sec)
        if md.duration_seconds is not None:
            stop = min(stop, float(md.duration_seconds))

        # 기존 코드처럼 target_frames로 시간 균등 샘플링:
        # TorchCodec의 time 기반 API로 원하는 프레임만 바로 가져옵니다. :contentReference[oaicite:3]{index=3}
        n = int(self._target_frames)
        if n <= 0:
            raise ValueError(f"_target_frames must be > 0, got {self._target_frames}")

        if n == 1:
            ts = torch.tensor([start], dtype=torch.float64)
        else:
            span = max(stop - start, 1e-6)
            step = span / n  # half-open [start, stop) 느낌으로
            ts = start + step * torch.arange(n, dtype=torch.float64)
            ts = torch.clamp(ts, max=stop - 1e-6)

        frame_batch = decoder.get_frames_played_at(ts)  # FrameBatch :contentReference[oaicite:4]{index=4}
        v = frame_batch.data  # (T, C, H, W)

        # resize (torchvision.transforms.Resize 대신 interpolate로 명확하게 처리)
        t, c, h, w = v.shape
        h_pad = int(h * self._video_resize_rate)
        w_pad = int(w * self._video_resize_rate)
        if (h_pad, w_pad) != (h, w):
            v = F.interpolate(v, size=(h_pad, w_pad), mode="bilinear", align_corners=False)

        # 혹시라도 디코더가 예상보다 적게 반환한 경우 패딩(방어 코드)
        if t < n:
            v_pad = torch.zeros(size=(n, c, h_pad, w_pad), dtype=v.dtype)
            v_pad[:t] = v
            v = v_pad
        elif t > n:
            # 방어적으로 target_frames로 맞춤(보통은 발생하지 않습니다)
            idx = torch.linspace(0, t - 1, steps=n).round().long()
            v = v.index_select(0, idx)

        info: dict[str, Any] = {
            "video_fps": md.average_fps,
            "video_path": video_path,
            "start_sec": start,
            "end_sec": stop,
            "num_frames_total": md.num_frames,
            "duration_seconds": md.duration_seconds,
            "width": md.width,
            "height": md.height,
            "codec": md.codec,
            # 실제로 뽑힌 프레임들의 시간 정보
            "pts_seconds": frame_batch.pts_seconds,
            "frame_duration_seconds": frame_batch.duration_seconds,
        }
        return v, info

    def _load_audio(self, video_path: str, *, start_sec: float, end_sec: float) -> torch.Tensor:
        """
        mp4에서 오디오를 sample_rate로 디코딩 후, 시간 구간으로 잘라 (L,)로 반환
        """
        with fsspec.open(video_path, "rb") as f:
            dec = AudioDecoder(f)
            sample_rate = dec.metadata.sample_rate
            x = dec.get_all_samples().data  # Tensor
        if x.ndim == 2:
            # (C,L) 또는 (L,C) 케이스를 mono로 단순화
            if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
                x = x.mean(dim=0)
            else:
                x = x.mean(dim=1)
        x = x.to(torch.float32).contiguous().view(-1)

        s = max(int(round(float(start_sec) * sample_rate)), 0)
        e = min(int(round(float(end_sec) * sample_rate)), len(x))
        if e <= s:
            raise f"Current range has no audio ({start_sec}, {end_sec})"

        audio_meta = {**dec.metadata.__dict__}
        audio_meta.pop("sample_rate")

        return x[s:e], {"sampling_rate": sample_rate, **audio_meta}

    def _load_text(self, it: SampleIndex, sample_fps: float) -> tuple[Optional[Any], Optional[str]]:
        label_path = it.label_path
        fps = sample_fps if sample_fps is not None else 30
        start_frame = int(round(it.chunk.start_sec * fps))
        end_frame = int(round(it.chunk.end_sec * fps))

        if label_path is None:
            return None, None
        with open(label_path, "r", encoding="utf-8") as f:
            label_dict = json.load(f)

        data = label_dict["data"]
        sorted_k = sorted(data.keys(), key=lambda x: int(x), reverse=False)

        text_list, label_list = [], []
        for k in sorted_k:
            if not (start_frame <= int(k) <= end_frame):
                continue
            curr_data_dict = label_dict["data"][k]
            for curr_frame, curr_data in curr_data_dict.items():
                if "text" in curr_data:
                    text_list.append(curr_data["text"]["script"])
                if curr_data.get("emotion"):
                    label_list.append(curr_data["emotion"]["image"]["emotion"])
                else:
                    label_list.append("null")

        # get last valid text & label
        text_out = text_list[-1] if len(text_list) > 0 else "No Script"

        label: str = "null"
        for label in label_list[::-1]:
            if label != "null":
                break

        return text_out, LabelIndexMap[label]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self._data[idx]

        v, video_meta = self._load_video(it.video_path, start_sec=it.chunk.start_sec, end_sec=it.chunk.end_sec)
        a, audio_meta = self._load_audio(it.video_path, start_sec=it.chunk.start_sec, end_sec=it.chunk.end_sec)
        text, label = self._load_text(it, video_meta["video_fps"])

        return {
            "videos": v,  # raw frames (T,H,W,C) uint8
            "audio": a,  # waveform (L, ) float32
            "text": text,  # optional
            "labels": label,  # optional
            "metadata": {
                **video_meta,
                **audio_meta,
                "video_size": v.size() if isinstance(v, torch.Tensor) else len(v),
                "audio_size": a.size(),
                "text_len": len(text) if text else 0,
                "start_sec": it.chunk.start_sec,
                "end_sec": it.chunk.end_sec,
            },
        }


if __name__ == "__main__":
    ds = AIHubMultimodalDataset(
        data_dir="../../../assets/aihub/2019-01-005.멀티모달영상_sample/원천데이터",
        label_dir="../../../assets/aihub/2019-01-005.멀티모달영상_sample/라벨데이터",
        data_extension="mp4",
        label_extension="json",
        label_prefix="clip_",
        start_video_sec=0.0,
        end_video_sec=60.0,
        video_chunk_sec=2,
        fps=4,
        video_resize_rate=0.2,
        require_label=False,
    )
    data = ds[7]

    from transformers import (
        WhisperFeatureExtractor,
        AutoTokenizer,
        AutoImageProcessor,
        AutoProcessor,
        AutoVideoProcessor,
    )
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

    data["videos"]
    data["audio"]
    data["text"]
    video_processor = Qwen2VLVideoProcessor(merge_size=4)
    audio_processor = WhisperFeatureExtractor(chunk_length=4, sampling_rate=data["metadata"]["sampling_rate"])
    tokenizer = AutoTokenizer.from_pretrained("yanolja/YanoljaNEXT-EEVE-2.8B")

    videos = video_processor(data["videos"])
    audio = audio_processor(data["audio"])
    text = tokenizer(data["text"])
    print(1)

    AutoVideoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B", use_fast=True)
    ps = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
