# Multimodal Training Framework (Lightning + Hydra + 🤗 Transformers)

A **configuration-driven, composable framework** for training **multimodal classification models** (video / audio / text).

The goal of this repository is **not** to ship a specific SOTA model as-is, but to provide a **multimodal training skeleton** where:

- experiment / architecture combinations are **declared in YAML (Hydra)**,
- the training loop is standardized with **PyTorch Lightning**, and
- model components follow the **🤗 Transformers `PreTrainedModel` / `PretrainedConfig` conventions**,

so that **swapping components is easy**.

> The example data/configuration is written against an **AIHub multimodal video sample** dataset (mp4 + label json).

---

## Problems this framework is designed to solve

### 1) Avoid editing code per experiment
In multimodal training, many things change from run to run:

- you want to switch the video encoder from CNN to ViT,
- you want to drop audio/text or add a new modality,
- you want to replace fusion (concat → bottleneck/transformer),
- you want to change the head (number of classes / task).

This repo is designed so that these changes can be done **via YAML composition/overrides, without touching code**.

### 2) Separate “training infra” from “serving artifacts”
Training is convenient with Lightning, while serving/deployment is often convenient in HF’s `save_pretrained()` format.

- Training: the LightningModule handles loss / logging / optimizer / scheduler
- Model architecture: the HF `PreTrainedModel` assembles (encoder/fusion/head)
- Config & reproducibility: HF `PretrainedConfig` + Hydra

In short, the repo separates **training infrastructure** from **model composition**.

### 3) Safety rails for dynamic imports
Pointing to classes via YAML `_target_` is powerful, but importing arbitrary modules can be risky.

- This repo supports `allowed_target_prefixes` to restrict what can be imported,
  improving reproducibility and safety.

---

## What’s inside (implemented components)

### Entry point
- `train.py`
  - loads Hydra config and sets seeds
  - assembles **HF Config → HF Model → LightningModule**
  - runs Lightning `Trainer.fit()`

### Configuration system (Hydra)
- `config/base.yaml`
  - Trainer / Logger(TensorBoard) / Callbacks(EarlyStopping, Checkpoint, LR monitor)
  - creates an experiment directory like `outputs/YYYY-MM-DD/HH-MM-SS/`
- `config/module/*`
  - LightningModule composition (model / optimizer / scheduler / loss)
- `config/datamodule/*`
  - Dataset(Train/Val/Test/Predict) + Processor (= collate_fn)
- `config/hf_config/*`
  - declares encoder / fusion / head via `_target_`

### Training loop (Lightning)
- `src/module/lit_module.py` (`LitMultimodalModule`)
  - computes loss in `training_step` / `validation_step` using `outputs.logits` and `labels`
  - injects optimizer/scheduler from config via `configure_optimizers()`
  - currently, the `save_pretrained()` hook in `on_train_end()` is TODO

### Model assembly (HF Config/Model)
- `src/module/model/hf_config.py` (`MultimodalPreTrainedConfig`)
  - holds `encoders`, `fusion`, `head` as **serializable dict specs**
  - supports `allowed_target_prefixes`
  - (optional) supports `id2label`, `label2id`
- `src/module/model/hf_model.py` (`MultimodalPreTrainedModel`)
  - reads specs from config and assembles **encoders → fusion → head** at runtime
  - forward signature follows HF style
    - text: `input_ids`, `attention_mask`
    - vision: `pixel_values_videos`
    - audio: `input_features`, `audio_attention_mask`
  - does **not** compute loss (returns logits only); LightningModule handles loss
- `src/module/utils/import_util.py`
  - minimal `_target_` instantiation utility (works without Hydra)
  - includes prefix allowlist checks

### Encoders
- `SimpleCNN1DEncoder` (`src/module/model/encoders/common/simple.py`)
  - input: `(B, T, D_in)`
  - linear projection + 1D conv blocks (over time axis)
  - used as a default for video/audio
- `EEVEEmbedddingEncoder` (`src/module/model/encoders/text/eeve2_5_embedding.py`)
  - loads **token embeddings only** from `yanolja/YanoljaNEXT-EEVE-2.8B` (HF Hub),
    then projects to `out_features` with a Linear layer
  - input: `input_ids (B, L)` → output: `(B, L, C)`
- `DummyEncoder`
  - testing/dummy purposes (current output shape does not match normal training; see limitations)

### Fusion
- `SimpleTimeConcat` (`src/module/model/fusion/concat.py`)
  - concatenates per-modality `(B, T, C)` on the time axis → pool → MLP
  - useful for fast baselines / debugging
- `MultimodalBottleneckFusion` (`src/module/model/fusion/multimodal_bottleneck.py`)
  - MBT-style: attaches shared bottleneck tokens to per-modality Transformer blocks,
    exchanging information across layers
  - supports **per-modality RoPE** (speed tuning via `rope_factor`)
  - includes a structure that multiplies a **gate** into SDPA outputs to mitigate attention sinks
  - output: flattened bottleneck vector `(B, num_bottleneck_tokens * embed_dim)`

> `fusion/transformer.py` and `encoders/vision/qwen2vl_resampler.py` are currently TODO (placeholders only).

### Head
- `SimpleLinear` (`src/module/model/head/linear.py`)
  - converts the fused vector to class logits

### Dataset / DataLoader
- `AIHubMultimodalDataset` (`src/datamodule/datasets/aihub_multimodal.py`)
  - reads mp4 files + label json and creates **chunk-level samples**
  - decoding via `torchcodec`
    - Video: uniform sampling within a segment via `VideoDecoder.get_frames_played_at()`
    - Audio: `AudioDecoder.get_all_samples()` then slice the segment
  - extracts text / emotion for the segment from label json
  - output (raw):

    ```python
    {
      "videos": Tensor(T, C, H, W),
      "audio": Tensor(L,),
      "text": str | None,
      "labels": int | None,
      "metadata": {...},
    }
    ```

- `LitMultimodalDataModule` (`src/datamodule/lit_datamodule.py`)
  - wraps datasets into DataLoaders and builds batches via Processor (collate_fn)

### Processor (Collate)
- `SimpleMultimodalProcessor` (`src/datamodule/processors/hf_processor.py`)
  - built on HF `ProcessorMixin`
  - default components (replaceable via config):
    - video_processor: `Qwen2VLVideoProcessor`
    - feature_extractor: `WhisperFeatureExtractor`
    - tokenizer: `AutoTokenizer` (EEVE 2.8B)
  - runs per-sample preprocessing inside `collate_fn(batch)` and stacks into tensors
  - permutes audio features from `(B, C, T)` → `(B, T, C)` to match encoder input

---

## Quick start

### 1) Installation

There is no `requirements.txt` included. Below is a minimal example that should run:

```bash
pip install -U   torch torchvision torchaudio   lightning hydra-core omegaconf   transformers accelerate   torchcodec fsspec numpy   absl-py
```

- Default config for Apple Silicon (Mac) is `trainer.accelerator=mps`.
- On CUDA machines, override as shown below.

### 2) Data preparation (example: AIHub sample)

The default config expects the following paths:

- videos: `assets/aihub/2019-01-005.멀티모달영상_sample/원천데이터`
- labels: `assets/aihub/2019-01-005.멀티모달영상_sample/라벨데이터`

```text
assets/
  aihub/
    2019-01-005.멀티모달영상_sample/
      원천데이터/   (*.mp4)
      라벨데이터/   (clip_*.json)
```

### 3) Run training

From the repo root:

```bash
PYTHONPATH=./src python train.py
```

Hydra will create an output directory like:

```text
outputs/YYYY-MM-DD/HH-MM-SS/
  checkpoints/
  events.out.tfevents...   # TensorBoard
```

---

## Config structure (what changes what)

- `config/base.yaml`
  - Trainer / Logger / Callbacks / output directory
- `config/module/lit_module.yaml` + children
  - optimizer / scheduler / loss injected into LightningModule
- `config/datamodule/lit_datamodule.yaml` + children
  - Dataset/Processor + DataLoader parameters
- `config/hf_config/base.yaml` + children
  - encoder/fusion/head composition (= model architecture)

`train.py` assembles in this order:

1) create `MultimodalPreTrainedConfig` from `cfg.hf_config` (dict)
2) create `MultimodalPreTrainedModel(config)`
3) create `LitMultimodalModule(model=..., optimizer=..., scheduler=..., loss_func=...)`

---

## Common Hydra override examples

### Train on CUDA

```bash
PYTHONPATH=./src python train.py   trainer.accelerator=cuda trainer.devices=1   trainer.precision=16-mixed
```

### Change batch size / workers

```bash
PYTHONPATH=./src python train.py   datamodule.batch_size=8 datamodule.num_workers=8
```

### Change data paths

```bash
PYTHONPATH=./src python train.py   datamodule.train_dataset.data_dir=/path/to/videos   datamodule.train_dataset.label_dir=/path/to/labels
```

### Replace fusion (bottleneck → concat)

```bash
PYTHONPATH=./src python train.py hf_config/fusion=concat hf_config/head=linear
```

> When switching to concat, it’s safer to swap the head together because the head’s `in_features` must match.

### Replace text encoder with dummy (for debugging)

```bash
PYTHONPATH=./src python train.py hf_config/encoders/text=dummy
```

---

## How to extend

### 1) Add a new Encoder / Fusion / Head

1. Implement a `torch.nn.Module` under `src/module/model/...`
2. Add a YAML file under `config/hf_config/...` with `_target_` pointing to the class
3. If needed, add its prefix to `allowed_target_prefixes` in `config/hf_config/base.yaml`

Example:

```yaml
# config/hf_config/encoders/vision/my_encoder.yaml
_target_: module.model.encoders.vision.my_encoder.MyVisionEncoder
in_features: 123
out_features: 256
```

### 2) Add a new Dataset / Processor

- The Dataset can return raw items from `__getitem__`.
- The Processor should implement `collate_fn(batch) -> dict[str, Tensor|list]` and create batches that match the model’s expected inputs.

**Important:** The Processor must produce the input keys expected by the HF model `forward()`:

- text: `input_ids`, `attention_mask`
- vision: `pixel_values_videos`
- audio: `input_features`, `audio_attention_mask` (optional)

---

## Limitations / known issues

This repo focuses on a “framework skeleton”, so the following items are intentionally (or not yet) incomplete.

### Serving artifacts are not fully implemented
- `LitMultimodalModule.on_train_end()` is `pass`, so there is no `save_pretrained()` export hook.
- `SimpleMultimodalProcessor.save_pretrained()` is also TODO.

### Masking / padding support is incomplete
- `MultimodalBottleneckFusion` does not apply input masks to attention (assumes no token padding).
- Even if provided, `attention_mask` (text) and `audio_attention_mask` are not currently used inside encoder/fusion.

### Position-id caching in MBT fusion is fragile with variable lengths
- `MultimodalBottleneckFusion` caches position_ids by `(modality, bottleneck_start flag)`.
- If sequence lengths vary across batches, cached values may be reused incorrectly.

### Data loading performance
- Audio decoding uses `get_all_samples()` then slices the segment (may be inefficient for long videos).
- Video decoding opens a decoder per sample and reads frames without caching/prefetch.

### Placeholders and bug-prone parts
- `DummyEncoder` returns `(1, out_features)` regardless of batch size, making it unsuitable for real training.
- `fusion/transformer.py`, `encoders/vision/qwen2vl_resampler.py`, `config/module/loss_func/mse.yaml`, etc. are placeholders.
- `AIHubMultimodalDataset._load_audio()` contains `raise f"..."` (raising a string instead of an Exception); consider replacing with `ValueError(...)`.
- `src/module/lit_module.py` still imports `absl` (may fail if `absl-py` is not installed). It’s unused and can be removed.

### Preprocessing / shape alignment depends heavily on user configuration
- The tensor shape produced by video/audio processors (especially the last feature dim) must match the encoders’ `in_features`.
- The default `vision.in_features=1176` and `audio.in_features=80` are tuned to the current Processor output assumptions.
  If you replace processors, you likely need to adjust these values together.

---

## Troubleshooting

- **`torchcodec` / FFmpeg issues**: depending on your environment, additional installation steps may be required.
- **`pin_memory` on MPS**: default is `false` (`pin_memory` is not supported on `mps`).
- **HF model downloads**: `EEVEEmbedddingEncoder` downloads from Hugging Face Hub.
  - In restricted networks, pre-cache the model or swap to a different text encoder.

---

## License

Assumed to be internal/personal sample code. Add a license statement that matches your project policy if needed.
