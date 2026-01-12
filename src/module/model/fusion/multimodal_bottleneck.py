from typing import Dict, List, Optional, Literal, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor, device
from torch.nn import RMSNorm
from transformers import ROPE_INIT_FUNCTIONS, PretrainedConfig
from transformers.models.exaone4.modeling_exaone4 import apply_rotary_pos_emb

from module.model.encoders.base import ModalityName

PositionEmbeddingType = Literal["none", "learnable", "rope"]
BottleNeckPosPolicy = Literal["reset", "zeros"]
BottleNeckOutPolicy = Literal["last", "mean"]


class RoPE(nn.Module):
    def __init__(self, rope_config: PretrainedConfig, rope_type: str = "default", device: str = None):
        super().__init__()
        self.rope_type = rope_type
        self._rope_init_func = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, attention_scaling = self._rope_init_func(config=rope_config, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = attention_scaling

    @torch.no_grad()
    def forward(self, x_ref: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_ref: dtype/device 참고용 텐서 (예: q tensor)
        position_ids: (B, L) long
        returns: cos, sin = (B, L, rope_dim) with dtype=x_ref.dtype
        """
        if position_ids.dtype != torch.long:
            position_ids = position_ids.to(dtype=torch.long)

        b = position_ids.shape[0]
        device = x_ref.device
        device_type = x_ref.device.type if isinstance(x_ref.device.type, str) else "cpu"

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(b, -1, 1).to(device)  # 1, D/2, 1
        position_ids_expanded = position_ids[:, None, :].float()  # B, 1, L

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # (B, L, D/2; sin & cos)
            emb = torch.cat((freqs, freqs), dim=-1)  # (B, L, D)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x_ref.dtype), sin.to(dtype=x_ref.dtype)


# Learnable APE
class LearnablePE(nn.Module):
    def __init__(self, sequence_len: int, dim: int):
        super().__init__()
        self.max_len = int(sequence_len)
        self.emb = nn.Embedding(self.max_len, dim)

    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        if length > self.max_len:
            raise ValueError(f"length={length} exceeds max_len={self.max_len}")
        pos = torch.arange(length, device=device, dtype=torch.long)
        return self.emb(pos)[None, :, :]  # (1, L, C)


class SDPAOutputGate(nn.Module):
    """
    sigmoid gate to avoid attention sink (https://arxiv.org/abs/2505.06708)
    Currently, only headwise for granularity is available.

    - x: (B, L, C) # prenorm
    - attn_out:  (B, H, L, Dh)
    """

    def __init__(
        self, *, embed_dim: int, num_heads: int, head_dim: int, activation: Literal["sigmoid", "silu"] = "sigmoid"
    ):
        super().__init__()
        self.activation = activation
        self.proj = nn.Linear(embed_dim, num_heads, bias=True)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        b, h, l, dh = attn_out.shape
        if h != self.num_heads or dh != self.head_dim:
            raise ValueError(
                f"attn_out shape mismatch: got (H,Dh)=({h},{dh}), expected ({self.num_heads},{self.head_dim})"
            )

        gate_logits = self.proj(x)  # (B, L, out_dim)

        if self.activation == "sigmoid":
            gate = torch.sigmoid(gate_logits)
        else:
            gate = F.silu(gate_logits)

        gate = gate.view(b, l, h, 1).permute(0, 2, 1, 3)  # (B,H,L,1)

        # Y * act(X@W)
        return attn_out * gate


# -------------------------
# Attention (RoPE + G1 gate)
# -------------------------
class GatedMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rotary_emb: Optional[RoPE] = None,
        gate_activation: Literal["sigmoid", "silu"] = "sigmoid",
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = float(dropout)

        self.proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rotary_emb = rotary_emb

        # G1 gate: SDPA 출력 직후
        self.gate = SDPAOutputGate(
            embed_dim=embed_dim, num_heads=num_heads, head_dim=self.head_dim, activation=gate_activation
        )

    def forward(self, x_prenorm: torch.Tensor, *, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_prenorm: (B, L, C)  (pre norm hidden)
        """
        b, l, c = x_prenorm.shape
        qkv = self.proj(x_prenorm)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, H, L, Dh)
        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE: Q,K에만 적용
        if self.rotary_emb is not None:
            if position_ids is None:
                raise ValueError("position_ids must be provided when pos_type='rope'.")
            assert self.rotary_emb is not None

            cos, sin = self.rotary_emb(q, position_ids)  # (B, L, rope_dim)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # SDPA
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )  # (B,H,L,Dh)

        # G1 gate
        attn_out = self.gate(x_prenorm, attn_out)  # (B,H,L,Dh)

        # (B,L,C) -> out_proj
        out = attn_out.transpose(1, 2).contiguous().view(b, l, c)
        return self.out_proj(out)


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, embed_dim: int, expansion_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * expansion_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RMSNormFP32(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # 정규화는 fp32로
        y = F.rms_norm(x.float(), self.normalized_shape, self.weight.float(), self.eps)
        # 출력은 입력 dtype으로 복귀
        return y.to(orig_dtype)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rotary_emb: Optional[RoPE] = None,
        norm_eps: float = 1e-6,
        # gate options
        gate_activation: Literal["sigmoid", "silu"] = "sigmoid",
    ):
        super().__init__()
        self.norm1 = RMSNormFP32(embed_dim, eps=norm_eps)
        self.attn = GatedMultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            rotary_emb=rotary_emb,
            gate_activation=gate_activation,
        )
        self.norm2 = RMSNormFP32(embed_dim, eps=norm_eps)
        self.ffn = FeedForwardSwiGLU(embed_dim, expansion_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, *, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), position_ids=position_ids)
        x = x + self.ffn(self.norm2(x))
        return x


# -------------------------
# MBT Fusion (모달별 RoPE, 각속도는 rope_factor로 조정)
# -------------------------
class MultimodalBottleneckFusion(nn.Module):
    """
    Multimodal Bottleneck Fusion (MBT 스타일)

    입력:
      x: Dict[str, Tensor], 각 Tensor는 (B, Tm, C)
      - padding은 데이터셋에서 처리됐다고 가정 (mask 없음)

    동작:
      각 레이어에서 모달별로 [tokens_m || bottleneck]을 TransformerBlock으로 통과시키고,
      각 모달에서 나온 bottleneck을 평균내어 공유 bottleneck을 갱신.

    position_embedding_type:
      - "none": 위치 신호 없음
      - "learned": 입력에 LearnablePE를 더함
      - "rope": attention 내부에서 RoPE 적용 (모달별 RoPE)

    bn_pos_policy (rope에서만 의미):
      - "reset": bottleneck 위치를 0..Bn-1로
      - "zeros": bottleneck 위치를 전부 0으로

    출력:
      기본: (B, C)  (bottleneck 토큰 평균)
      return_bottlenecks=True면 (B, C), (B, Bn, C) 반환
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_bottleneck_tokens: int = 4,
        dropout: float = 0.0,
        modality_names: Optional[List[str]] = None,
        position_embedding_type: PositionEmbeddingType = "rope",
        bottleneck_start_index: int = 3,
        bottleneck_pos_policy: BottleNeckPosPolicy = "reset",
        bottleneck_out_policy: BottleNeckOutPolicy = "last",
        # learned pe
        learned_max_seq_len: int = 4096,
        use_bottleneck_learned_pos: bool = True,
        # rope for each modality (dict)
        rope_config_dict: Optional[dict[str, dict[str, Any]]] = None,
        # gate activation forwarded to TransformerBlock
        gate_activation: Literal["sigmoid", "silu"] = "sigmoid",
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        head_dim = embed_dim // num_heads

        if modality_names is None:
            modality_names = list(ModalityName.__args__)
        self.modality_names = modality_names

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens
        self.position_embedding_type = position_embedding_type
        self.bottleneck_start_index = bottleneck_start_index
        self.bottleneck_pos_policy = bottleneck_pos_policy
        self.bottleneck_out_policy = bottleneck_out_policy

        # shared bottleneck tokens
        self.bottleneck = nn.Parameter(torch.randn(1, num_bottleneck_tokens, embed_dim) * 0.02)
        self.bottlenect_pe: nn.Module = None

        # position embedding
        match position_embedding_type:
            case "learnable":
                self.pe_dict = nn.ModuleDict({m: LearnablePE(learned_max_seq_len, embed_dim) for m in modality_names})
                self.bottlenect_pe = (
                    LearnablePE(num_bottleneck_tokens, embed_dim) if use_bottleneck_learned_pos else None
                )

            case "rope":
                self.rope_dict = nn.ModuleDict()
                for m in modality_names:
                    rope_config = PretrainedConfig(**rope_config_dict[m])
                    rope = RoPE(rope_config=rope_config, device=None)
                    rope_factor = rope_config.rope_factor
                    rope.inv_freq.div_(rope_factor)
                    self.rope_dict[m] = rope

            case _:
                self.pe_dict = None

        self._pos_cache: dict[tuple[str, bool], torch.Tensor] = {}

        # modality-wise transformer stacks
        # self.blocks = nn.ModuleDict()
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        m: TransformerBlock(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            dropout=dropout,
                            rotary_emb=self.rope_dict[m] if position_embedding_type == "rope" else None,
                            gate_activation=gate_activation,
                        )
                        for m in modality_names
                    }
                )
            )

    def build_position_ids(
        self,
        batch_size: int,
        modality_t: int,
        bottleneck_t: int,
        device: torch.device,
    ) -> torch.Tensor:
        mod_pos = torch.arange(modality_t, device=device, dtype=torch.long)
        if bottleneck_t == 0:
            pos = mod_pos
        else:
            bottleneck_pos_policy = self.bottleneck_pos_policy
            if bottleneck_pos_policy == "zeros":
                bn_pos = torch.zeros(bottleneck_t, device=device, dtype=torch.long)
            else:
                bn_pos = torch.arange(bottleneck_t, device=device, dtype=torch.long)

            pos = torch.cat([mod_pos, bn_pos], dim=0)  # (t_mod + t_bn,)
        return pos[None, :].expand(batch_size, -1)  # (B, L)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        (psuedo codes)
        # cross modal information exchange after modality specific layers
        modality_A(A_tokens, Learnable)
        modality_B(B_tokens, Learnable)
        ...
        return Learnable (or CLS)
        """

        modality_list = [m for m in self.modality_names if m in x]
        if len(modality_list) == 0:
            raise ValueError("No modalities found in input dict x.")

        input_embedding_dict, meta = self._get_input_embedding_dict(modality_list, x)
        b, device = meta["batch_size"], meta["device"]

        # init bottleneck
        bn = self._init_bottleneck(b, device)
        # MBT layers
        bn = self._forward_bottleneck(b, bn, device, input_embedding_dict, modality_list)
        return bn

    def _forward_bottleneck(
        self, b: int, bn: Tensor, device: torch.device, input_embedding_dict: int, modality_list: list[str]
    ) -> torch.Tensor:
        bn_t = bn.size(1)

        for enum, layer in enumerate(self.blocks):
            bn_output_list: list[torch.Tensor] = []
            is_bottleneck_start = self.bottleneck_start_index <= enum

            for m in modality_list:
                modality_embedding = input_embedding_dict[m]  # (B, Tm, C)
                modality_t = modality_embedding.size(1)

                if is_bottleneck_start:
                    fused_seq = torch.cat([modality_embedding, bn], dim=1)  # (B, Tm+Bn, C)
                else:
                    fused_seq = modality_embedding

                pos_cache_key = (m, is_bottleneck_start)
                if self.position_embedding_type == "rope":
                    pos_cache = self._pos_cache.get(pos_cache_key, None)
                    if pos_cache is None:
                        position_ids = self.build_position_ids(
                            batch_size=b,
                            modality_t=modality_t,
                            bottleneck_t=bn_t if is_bottleneck_start else 0,
                            device=device,
                        )
                        self._pos_cache[pos_cache_key] = position_ids
                    else:
                        position_ids = self._pos_cache[pos_cache_key]
                else:
                    position_ids = None

                out = layer[m](fused_seq, position_ids=position_ids)
                input_embedding_dict[m] = out[:, :modality_t, :]

                if is_bottleneck_start:
                    # bottleneck for next layer
                    bn_output_list.append(out[:, modality_t:, :])  # B, Bn, C

            if is_bottleneck_start:
                if self.bottleneck_out_policy == "last":
                    bn = bn_output_list[-1]
                elif self.bottleneck_out_policy == "mean":
                    bn = torch.stack(bn_output_list, dim=0).mean(dim=0)
        return bn.view(b, -1)

    def _init_bottleneck(self, b, device) -> Tensor:
        bn = self.bottleneck.expand(b, -1, -1).to(device=device)  # (B, Bn, C)
        if self.position_embedding_type == "learnable":
            bn = bn + self.bottlenect_pe(length=self.num_bottleneck_tokens, device=device)
        return bn

    def _get_input_embedding_dict(self, present: list[str], x: dict[str, Tensor]) -> tuple[int, dict]:
        input_embedding_dict: dict[str, torch.Tensor] = {}
        b: int = None
        device: torch.device = torch.device("cpu")
        # validate + (optional) learned PE
        for m in present:
            modality_embedding = x[m]
            if modality_embedding.size(-1) != self.embed_dim:
                raise ValueError(
                    f"Expected embed_dim={self.embed_dim}, got x[{m}].size(-1)={modality_embedding.size(-1)}"
                )

            b, t, c = modality_embedding.shape
            device = modality_embedding.device

            if self.position_embedding_type == "learnable":
                modality_embedding = modality_embedding + self.pe_dict[m](t, device=device)

            input_embedding_dict[m] = modality_embedding
        return input_embedding_dict, {"batch_size": b, "device": device}


if __name__ == "__main__":
    # ROPE_CONFIG = PretrainedConfig(rope_theta=10000.0, hidden_size=256, num_attention_heads=4, rope_factor=1.0)
    # ROPE_INIT_FUNCTIONS["default"](ROPE_CONFIG, device="mps")

    rope_config_dict = {
        "vision": {"rope_theta": 10000.0, "hidden_size": 256, "num_attention_heads": 4, "rope_factor": 8.0},
        "audio": {"rope_theta": 10000.0, "hidden_size": 256, "num_attention_heads": 4, "rope_factor": 4.0},
        "text": {"rope_theta": 10000.0, "hidden_size": 256, "num_attention_heads": 4, "rope_factor": 1.0},
    }

    fusion = MultimodalBottleneckFusion(
        embed_dim=256,
        num_layers=6,
        num_heads=4,
        num_bottleneck_tokens=4,
        # position_embedding_type="rope",
        position_embedding_type="learnable",
        rope_config_dict=rope_config_dict,
        bottleneck_start_index=2,
        bottleneck_out_policy="last",
        bottleneck_pos_policy="zeros",
        gate_activation="sigmoid",
    )

    fusion({m: torch.randn(size=(1, 32, 256)) for m in ["vision", "audio", "text"]})
