"""
Qwen 2.5/3 Numpy Prototype (Reference Implementation)

This script provides a pure Python/Numpy implementation of the Qwen 3 architecture.
It includes a native GGUF v3 reader and supports 'fused' quantized inference,
meaning it operates directly on quantized weights (Q8_0, Q8_K) without fully
dequantizing them to Float32 memory, mimicking how a high-performance GPU kernel works.

Key Features:
- Native GGUF Reader (no external C dependencies for loading).
- Fused Block-Quantized Matrix Multiplication.
- Qwen 3 specific architecture (Q-Norm, K-Norm, NEOX-style RoPE).
- KV Cache with pre-allocation.
"""

import argparse
import mmap
import os
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import llama_cpp
import numpy as np

# -----------------------------------------------------------------------------
# GGUF Constants & Types
# -----------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# Value types
GGUF_VALUE_TYPE_UINT8 = 0
GGUF_VALUE_TYPE_INT8 = 1
GGUF_VALUE_TYPE_UINT16 = 2
GGUF_VALUE_TYPE_INT16 = 3
GGUF_VALUE_TYPE_UINT32 = 4
GGUF_VALUE_TYPE_INT32 = 5
GGUF_VALUE_TYPE_FLOAT32 = 6
GGUF_VALUE_TYPE_BOOL = 7
GGUF_VALUE_TYPE_STRING = 8
GGUF_VALUE_TYPE_ARRAY = 9
GGUF_VALUE_TYPE_UINT64 = 10
GGUF_VALUE_TYPE_INT64 = 11
GGUF_VALUE_TYPE_FLOAT64 = 12

# Tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_I8 = 16
GGML_TYPE_I16 = 17
GGML_TYPE_I32 = 18

# -----------------------------------------------------------------------------
# Mini GGUF Reader
# -----------------------------------------------------------------------------
# Implements a parser for the GGUF v3 binary format.
# Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
#
# The file layout is:
# 1. Header (Magic, Version, Counts)
# 2. Key-Value Pairs (Metadata like dimensions, vocab, RoPE freq)
# 3. Tensor Info Table (Name, Dims, Type, Offset)
# 4. Padding (Alignment)
# 5. Tensor Data (Binary blobs)


class GGUFReader:
    """
    Reads GGUF files using memory mapping for efficiency.
    Does not load tensor data into memory until get_tensor() is called.
    """

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0

        self.kv_data = {}
        self.tensors = {}  # name -> { 'type', 'dims', 'offset' }
        self.data_offset = 0

        self._load()

    def _read(self, n):
        data = self.mm[self.pos : self.pos + n]
        self.pos += n
        return data

    def _read_unpack(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self._read(size))

    def _read_string(self):
        (length,) = self._read_unpack("<Q")
        return self._read(length).decode("utf-8")

    def _read_value(self, vtype):
        if vtype == GGUF_VALUE_TYPE_UINT8:
            return self._read_unpack("<B")[0]
        elif vtype == GGUF_VALUE_TYPE_INT8:
            return self._read_unpack("<b")[0]
        elif vtype == GGUF_VALUE_TYPE_UINT16:
            return self._read_unpack("<H")[0]
        elif vtype == GGUF_VALUE_TYPE_INT16:
            return self._read_unpack("<h")[0]
        elif vtype == GGUF_VALUE_TYPE_UINT32:
            return self._read_unpack("<I")[0]
        elif vtype == GGUF_VALUE_TYPE_INT32:
            return self._read_unpack("<i")[0]
        elif vtype == GGUF_VALUE_TYPE_FLOAT32:
            return self._read_unpack("<f")[0]
        elif vtype == GGUF_VALUE_TYPE_BOOL:
            return self._read_unpack("<?")[0]
        elif vtype == GGUF_VALUE_TYPE_STRING:
            return self._read_string()
        elif vtype == GGUF_VALUE_TYPE_UINT64:
            return self._read_unpack("<Q")[0]
        elif vtype == GGUF_VALUE_TYPE_INT64:
            return self._read_unpack("<q")[0]
        elif vtype == GGUF_VALUE_TYPE_FLOAT64:
            return self._read_unpack("<d")[0]
        elif vtype == GGUF_VALUE_TYPE_ARRAY:
            (vtype_len,) = self._read_unpack("<I")  # Array type (first is type)
            (length,) = self._read_unpack("<Q")
            return [self._read_value(vtype_len) for _ in range(length)]
        else:
            raise ValueError(f"Unknown value type: {vtype}")

    def _load(self):
        # 1. Header
        # Magic "GGUF" (4 bytes)
        magic = self._read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic}")
        (version,) = self._read_unpack("<I")
        (tensor_count,) = self._read_unpack("<Q")
        (kv_count,) = self._read_unpack("<Q")

        print(f"GGUF v{version} | Tensors: {tensor_count} | KV: {kv_count}")

        # KV Pairs
        for _ in range(kv_count):
            key = self._read_string()
            (vtype,) = self._read_unpack("<I")
            val = self._read_value(vtype)
            self.kv_data[key] = val

        # Tensor Info
        for _ in range(tensor_count):
            name = self._read_string()
            (n_dims,) = self._read_unpack("<I")
            dims = [self._read_unpack("<Q")[0] for _ in range(n_dims)]
            (dtype,) = self._read_unpack("<I")
            (offset,) = self._read_unpack("<Q")
            self.tensors[name] = {"type": dtype, "dims": dims, "offset": offset}

        # Calculate data alignment
        alignment = self.kv_data.get("general.alignment", 32)

        # Align position to data start
        padding = (alignment - (self.pos % alignment)) % alignment
        self.pos += padding
        self.data_offset = self.pos

        print(f"Data offset: {self.data_offset}")

    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        if name not in self.tensors:
            return None

        meta = self.tensors[name]
        offset = self.data_offset + meta["offset"]
        dims = meta["dims"]  # GGUF stores [fastest, ..., slowest] -> Fortran order?
        # Actually GGUF conventions: ne[0] is X, ne[1] is Y.
        # Numpy wants (Y, X) usually (C-order).
        # So we reverse dims for shape.
        shape = tuple(dims[::-1])

        dtype_id = meta["type"]

        # Number of items (elements)
        num_elem = 1
        for d in dims:
            num_elem *= d

        raw_ptr = self.mm

        if dtype_id == GGML_TYPE_F32:
            arr = np.frombuffer(
                raw_ptr, dtype=np.float32, count=num_elem, offset=offset
            )
            return arr.reshape(shape).copy()

        elif dtype_id == GGML_TYPE_F16:
            arr = np.frombuffer(
                raw_ptr, dtype=np.float16, count=num_elem, offset=offset
            )
            return arr.astype(np.float32).reshape(shape).copy()

        elif dtype_id == GGML_TYPE_Q8_0:
            # Q8_0 Layout:
            # Block Size: 32 elements.
            # Block Bytes: 34 bytes.
            # Structure: [Delta (float16, 2 bytes)] + [Quants (int8, 32 bytes)]
            #
            # We parse this into two separate arrays:
            # - scales: [n_blocks] (float32)
            # - quants: [n_blocks, 32] (int8)

            block_size = 32
            block_stride = 34
            num_blocks = num_elem // block_size
            raw_bytes = raw_ptr[offset : offset + num_blocks * block_stride]

            # View as [Blocks, Stride] bytes to slice columns
            data_view = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
                num_blocks, block_stride
            )
            # Scales: Bytes 0-1 are f16
            scales = (
                np.frombuffer(data_view[:, 0:2].flatten(), dtype=np.float16)
                .astype(np.float32)
                .copy()
            )
            # Quants: Bytes 2-34 are int8
            quants = (
                np.frombuffer(data_view[:, 2:].flatten(), dtype=np.int8)
                .reshape(num_blocks, block_size)
                .copy()
            )

            # Reshape to logical dims [Out, In_blocks] for scales and [Out, In_blocks, BlockSize] for quants
            # We assume dim[-1] is the one that was quantized (Input dimension)
            # shape is (Out, In) usually
            out_dim = shape[0]
            in_blocks = shape[1] // block_size
            scales = scales.reshape(out_dim, in_blocks)
            quants = quants.reshape(out_dim, in_blocks, block_size)

            return QuantizedTensor(quants, scales, block_size)

        elif dtype_id == GGML_TYPE_Q8_K:
            # Q8_K Layout:
            # Block Size: 256 elements.
            # Block Bytes: 260 bytes.
            # Structure: [Delta (float32, 4 bytes)] + [Quants (int8, 256 bytes)]
            #
            # Note: The usage of float32 for delta distinguishes it from Q8_0.
            # This is the primary format used in "Q8_K_XL" GGUF files.

            block_size = 256
            block_stride = 260
            num_blocks = num_elem // block_size
            raw_bytes = raw_ptr[offset : offset + num_blocks * block_stride]

            data_view = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
                num_blocks, block_stride
            )
            scales = (
                np.frombuffer(data_view[:, 0:4].flatten(), dtype=np.float32)
                .astype(np.float32)
                .copy()
            )
            quants = (
                np.frombuffer(data_view[:, 4:].flatten(), dtype=np.int8)
                .reshape(num_blocks, block_size)
                .copy()
            )

            out_dim = shape[0]
            in_blocks = shape[1] // block_size
            scales = scales.reshape(out_dim, in_blocks)
            quants = quants.reshape(out_dim, in_blocks, block_size)

            return QuantizedTensor(quants, scales, block_size)

        else:
            print(
                f"Warning: Tensor {name} has unsupported type {dtype_id}. Returning zeros."
            )
            return np.zeros(shape, dtype=np.float32)


# -----------------------------------------------------------------------------
# Quantization Support
# -----------------------------------------------------------------------------


class QuantizedTensor:
    """
    Holds weights in a split structure (quants + scales) to support fused operations.
    - q: The int8 quantized values [Out, In_Blocks, BlockSize]
    - s: The float32 block scales [Out, In_Blocks]

    This avoids expanding the entire weight matrix to float32 (which would be 4GB+ for 1B params),
    keeping it compact (~1GB).
    """

    def __init__(self, quants, scales, block_size):
        self.q = quants
        self.s = scales
        self.block_size = block_size
        self.out_dim = quants.shape[0]
        self.in_dim = quants.shape[1] * block_size
        self.shape = (self.out_dim, self.in_dim)

    @property
    def T(self):
        # Logical transpose for x @ w.T syntax
        return self


def matmul_fused(x: np.ndarray, w: Union[np.ndarray, QuantizedTensor]) -> np.ndarray:
    """
    Computes x @ w.T using a fused kernel approach for quantized weights.

    Standard Approach: Dequantize W -> F32 [Out, In], then compute x @ W.T.
    Fused Approach:
      1. Reshape x to [..., In_blocks, BlockSize].
      2. Dot product x_blocks against w.q (int8) -> Accumulate in F32.
      3. Multiply result by w.s (scales).
      4. Sum over blocks.

    This preserves memory bandwidth and mimics GPU quantized kernels.
    """
    if isinstance(w, np.ndarray):
        return x @ w.T

    # Quantized Fused Matmul logic
    # x: [In] or [Batch, In]
    # w.q: [Out, In_blocks, BlockSize]
    # w.s: [Out, In_blocks]

    bs = w.block_size
    orig_shape = x.shape
    x_flat = x.flatten()

    # Verify dims
    if x_flat.shape[-1] != w.in_dim:
        raise ValueError(f"Shape mismatch: x {x.shape} vs w {w.shape}")

    # Reshape x to [In_blocks, BlockSize]
    # We assume x is contiguous for reshaping
    x_blocked = x_flat.reshape(-1, bs)

    # 1. Dot product with quants (int8 computation or float32 accumulation)
    # x_blocked: [In_blocks, BlockSize]
    # w.q:       [Out, In_blocks, BlockSize]
    # We want sum over BlockSize first.
    # Result: [Out, In_blocks]

    # Use float32 for accumulation to avoid overflow
    # np.einsum is efficient here
    # 'ib,oib->oi'
    dot_products = np.einsum(
        "ib,oib->oi", x_blocked.astype(np.float32), w.q.astype(np.float32)
    )

    # 2. Apply scales
    # w.s: [Out, In_blocks]
    scaled = dot_products * w.s

    # 3. Sum over blocks
    # Result: [Out]
    out = np.sum(scaled, axis=1)

    return out


# -----------------------------------------------------------------------------
# Qwen Model
# -----------------------------------------------------------------------------


class QwenConfig:
    def __init__(self, reader: GGUFReader):
        kv = reader.kv_data

        # Determine architecture (e.g. "llama", "qwen", "qwen3")
        arch = kv.get("general.architecture", "llama")
        print(f"Detected Architecture: {arch}")

        self.n_embd = kv.get(f"{arch}.embedding_length")
        self.n_layer = kv.get(f"{arch}.block_count")
        self.n_head = kv.get(f"{arch}.attention.head_count")

        if self.n_embd is None:
            raise ValueError(f"Missing n_embd. Available keys: {list(kv.keys())}")
        if self.n_layer is None:
            raise ValueError("Missing n_layer")
        if self.n_head is None:
            raise ValueError("Missing n_head")

        # Vocab size often implied by tokenizer array length
        if "tokenizer.ggml.tokens" in kv:
            self.n_vocab = len(kv["tokenizer.ggml.tokens"])
        else:
            self.n_vocab = kv.get(f"{arch}.vocab_size", 151936)

        self.n_ctx = kv.get(f"{arch}.context_length", 32768)

        self.n_head_kv = kv.get(f"{arch}.attention.head_count_kv", self.n_head)
        self.rope_freq_base = kv.get(f"{arch}.rope.freq_base", 1000000.0)
        self.rms_norm_eps = kv.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)

        self.head_dim = kv.get(f"{arch}.attention.key_length")
        if self.head_dim is None:
            self.head_dim = self.n_embd // self.n_head

        # Tokenizer flags
        self.add_bos = kv.get("tokenizer.ggml.add_bos_token", False)

        print(
            f"Config Loaded: D={self.n_embd}, L={self.n_layer}, H={self.n_head}, "
            f"KV={self.n_head_kv}, HeadDim={self.head_dim}, V={self.n_vocab}"
        )
        print(
            f"Meta: RoPE={self.rope_freq_base}, Eps={self.rms_norm_eps}, AddBOS={self.add_bos}"
        )


class QwenNumpyModel:
    def __init__(self, model_path: str):
        print(f"Loading GGUF: {model_path}")
        self.reader = GGUFReader(model_path)
        self.config = QwenConfig(self.reader)

        self.weights = {}
        self._load_weights()

    def _load_weights(self):
        print("Loading weights...")

        def load(name):
            # Fallback for old/new naming schemes
            if name in self.reader.tensors:
                return self.reader.get_tensor(name)

            # Map "blk.0" -> "layers.0" if needed, but usually GGUF uses standardized names now?
            # Actually standard GGUF is "blk.N" or "layers.N"?
            # Let's try standard names from the file list
            return None

        # 1. Embeddings
        self.weights["token_embd"] = load("token_embd.weight")
        self.weights["output_norm"] = load("output_norm.weight")
        self.weights["output"] = load("output.weight")
        if self.weights["output"] is None:
            self.weights["output"] = self.weights["token_embd"]

        # 2. Layers
        for i in range(self.config.n_layer):
            p = f"blk.{i}"

            # Try to determine prefix (blk.N vs layers.N)
            if f"blk.{i}.attn_norm.weight" in self.reader.tensors:
                p = f"blk.{i}"
            elif f"layers.{i}.attn_norm.weight" in self.reader.tensors:
                p = f"layers.{i}"

            self.weights[f"layer.{i}.attn_norm"] = load(f"{p}.attn_norm.weight")

            self.weights[f"layer.{i}.attn_q"] = load(f"{p}.attn_q.weight")
            self.weights[f"layer.{i}.attn_k"] = load(f"{p}.attn_k.weight")
            self.weights[f"layer.{i}.attn_v"] = load(f"{p}.attn_v.weight")
            self.weights[f"layer.{i}.attn_out"] = load(f"{p}.attn_output.weight")

            self.weights[f"layer.{i}.attn_q_norm"] = load(f"{p}.attn_q_norm.weight")
            self.weights[f"layer.{i}.attn_k_norm"] = load(f"{p}.attn_k_norm.weight")

            self.weights[f"layer.{i}.ffn_norm"] = load(f"{p}.ffn_norm.weight")
            self.weights[f"layer.{i}.ffn_gate"] = load(f"{p}.ffn_gate.weight")
            self.weights[f"layer.{i}.ffn_down"] = load(f"{p}.ffn_down.weight")
            self.weights[f"layer.{i}.ffn_up"] = load(f"{p}.ffn_up.weight")

            sys.stdout.write(f"\rLoaded layer {i + 1}/{self.config.n_layer}")
            sys.stdout.flush()
        print("\nWeights loaded.")

    def rms_norm(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        x_f32 = x.astype(np.float32)
        variance = np.mean(x_f32**2, axis=-1, keepdims=True)
        return (
            x_f32 * (1.0 / np.sqrt(variance + self.config.rms_norm_eps)) * w
        ).astype(x.dtype)

    def apply_rope(self, xq, xk, pos_idx):
        """
        Rotary Positional Embeddings (RoPE).
        Qwen/Llama use "NEOX" style rotation:
          Instead of adjacent pairs (-x1, x0), it splits the head dimension into two halves:
          [x_0...x_half | x_half...x_end]
          And rotates elements (x_i, x_{i+half}).
        """
        dim = self.config.head_dim

        # 1. Compute Theta
        # Standard RoPE formula: theta_i = base ^ (-2i / dim)
        theta = (
            1.0
            / (
                self.config.rope_freq_base
                ** (np.arange(0, dim, 2).astype(np.float32) / dim)
            )
        ).astype(np.float32)

        m = pos_idx
        freqs = np.outer(m, theta)  # [1, dim/2]
        cos_vals = np.cos(freqs)
        sin_vals = np.sin(freqs)

        def rotate(x):
            # x shape: [heads, dim]
            # Split into two halves (NEOX style)
            x = x.astype(np.float32)
            half = dim // 2
            x1 = x[..., :half]
            x2 = x[..., half:]

            # Apply rotation matrix
            x1_new = x1 * cos_vals - x2 * sin_vals
            x2_new = x1 * sin_vals + x2 * cos_vals
            return np.concatenate((x1_new, x2_new), axis=-1)

        return rotate(xq), rotate(xk)

    def forward(self, token_id: int, pos: int, kv_cache=None):
        """
        Runs the transformer forward pass for a single token.
        This includes:
        - Embedding Lookup
        - N Layers of:
            - RMSNorm (Pre-Norm)
            - QKV Projections (Fused)
            - QK-Norm (Qwen specific)
            - RoPE (Rotary Embedding)
            - KV Cache Update
            - GQA (Grouped Query Attention)
            - Softmax Attention
            - FFN (SwiGLU)
        - Final Norm & Classifier
        """
        # 1. Embedding
        # Directly lookup (copy) from embedding table (usually F32 or Q8)
        x = self.weights["token_embd"][token_id].copy()

        for i in range(self.config.n_layer):
            p = f"layer.{i}"
            residual = x.copy()

            # --- Attention Block ---

            # Pre-Norm
            x = self.rms_norm(x, self.weights[f"layer.{i}.attn_norm"])

            # QKV Projections (Fused Matmul)
            # x: [Dim]
            # wq/wk/wv: QuantizedTensor [Out, Dim]
            # Output: [Out]
            wq = self.weights[f"{p}.attn_q"]
            wk = self.weights[f"{p}.attn_k"]
            wv = self.weights[f"{p}.attn_v"]

            q = matmul_fused(x, wq)
            k = matmul_fused(x, wk)
            v = matmul_fused(x, wv)

            # Reshape for multi-head
            q = q.reshape(self.config.n_head, self.config.head_dim)
            k = k.reshape(self.config.n_head_kv, self.config.head_dim)
            v = v.reshape(self.config.n_head_kv, self.config.head_dim)

            # Q-Norm / K-Norm (Qwen 3 Feature)
            # RMSNorm applied to Q and K vectors before RoPE
            if self.weights.get(f"{p}.attn_q_norm") is not None:
                w_q_norm = self.weights[f"{p}.attn_q_norm"]
                w_k_norm = self.weights[f"{p}.attn_k_norm"]

                q = self.rms_norm(q, w_q_norm)
                k = self.rms_norm(k, w_k_norm)

            # RoPE
            q, k = self.apply_rope(q, k, pos)

            # KV Cache Management
            if kv_cache is not None:
                # Store current k/v in pre-allocated buffer
                kv_cache[i][0][pos] = k
                kv_cache[i][1][pos] = v

                # Retrieve history (0..pos)
                K_seq = kv_cache[i][0][: pos + 1]  # [T, Hkv, D]
                V_seq = kv_cache[i][1][: pos + 1]
            else:
                K_seq = k[np.newaxis, ...]
                V_seq = v[np.newaxis, ...]

            # GQA Broadcast (if n_head > n_kv_head)
            # Repeat K/V to match query heads
            n_rep = self.config.n_head // self.config.n_head_kv
            if n_rep > 1:
                K_seq = np.repeat(K_seq, n_rep, axis=1)
                V_seq = np.repeat(V_seq, n_rep, axis=1)

            # SDPA (Scaled Dot Product Attention)
            # q: [H, D]
            # K_seq: [T, H, D]
            # einsum "hd,thd->ht": dot product over D for each Head and Time
            scale = 1.0 / np.sqrt(self.config.head_dim)
            scores = np.einsum("hd,thd->ht", q, K_seq) * scale

            # Softmax
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            # Weighted Sum
            # probs: [H, T]
            # V_seq: [T, H, D]
            # einsum "ht,thd->hd": sum over T
            attn_out = np.einsum("ht,thd->hd", probs, V_seq)
            attn_out = attn_out.reshape(-1)

            # Output Projection
            x = matmul_fused(attn_out, self.weights[f"{p}.attn_out"])
            x += residual
            residual = x.copy()

            # --- FFN Block (SwiGLU) ---
            # Gate/Up projections -> Activation -> Down projection

            x = self.rms_norm(x, self.weights[f"layer.{i}.ffn_norm"])

            w_gate = self.weights[f"{p}.ffn_gate"]
            w_up = self.weights[f"{p}.ffn_up"]
            w_down = self.weights[f"{p}.ffn_down"]

            gate = matmul_fused(x, w_gate)
            up = matmul_fused(x, w_up)

            # Activation: Silu(gate) * up
            # Stable Sigmoid to prevent overflow in exp(-gate) for large negative values
            gate_f64 = gate.astype(np.float64)
            sigmoid = 1.0 / (1.0 + np.exp(-gate_f64))
            silu = gate * sigmoid.astype(gate.dtype)

            merged = silu * up
            x = matmul_fused(merged, w_down)

            x += residual

        # Final Norm & Classifier
        x = self.rms_norm(x, self.weights["output_norm"])
        logits = matmul_fused(x, self.weights["output"])
        return logits


def sample_logits(logits, temp=0.0):
    if temp == 0.0:
        return int(np.argmax(logits))

    # Softmax
    logits = logits.astype(np.float64)  # Ensure precision
    probs = np.exp(logits - np.max(logits))
    probs /= np.sum(probs)
    return int(np.random.choice(len(logits), p=probs))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(script_dir, "../../models/Qwen3-0.6B-UD-Q8_K_XL.gguf")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model, help="Path to Qwen GGUF file")
    parser.add_argument(
        "--prompt", default="User: Hello!\nAssistant:", help="Initial prompt"
    )
    parser.add_argument("--steps", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature")
    parser.add_argument(
        "--ref",
        action="store_true",
        help="Run reference llama.cpp generation for comparison",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    # 1. Initialize Tokenizer (vocab_only=True to avoid loading weights twice)
    print(f"Loading Tokenizer (llama.cpp) from {args.model}...")
    try:
        # vocab_only=True is critical to avoid allocating full GPU/CPU buffers for weights
        llm = llama_cpp.Llama(model_path=args.model, vocab_only=True, verbose=False)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    # 2. Initialize Numpy Model
    try:
        model = QwenNumpyModel(args.model)
    except Exception:
        import traceback

        traceback.print_exc()
        return

    # 3. Tokenize Prompt
    # Ensure raw bytes for encode
    prompt_bytes = args.prompt.encode("utf-8")

    # Respect model metadata for BOS
    add_bos = model.config.add_bos
    tokens = llm.tokenize(prompt_bytes, add_bos=add_bos)
    print(f"Prompt tokens (add_bos={add_bos}): {tokens}")

    # 4. Generation Loop
    # Pre-allocate KV Cache
    # Shape: [n_layers, 2 (K/V), max_seq, n_head_kv, head_dim]
    max_seq_len = len(tokens) + args.steps + 16  # padding
    kv_cache = []
    for _ in range(model.config.n_layer):
        # Use zeros. float32 to match weights
        k_cache = np.zeros(
            (max_seq_len, model.config.n_head_kv, model.config.head_dim),
            dtype=np.float32,
        )
        v_cache = np.zeros(
            (max_seq_len, model.config.n_head_kv, model.config.head_dim),
            dtype=np.float32,
        )
        kv_cache.append(
            (k_cache, v_cache)
        )  # Tuple or list is fine, code uses index 0/1

    print(f"\n--- Generating ({args.steps} steps) ---\n")
    print(args.prompt, end="", flush=True)

    curr_tokens = list(tokens)

    # Prefill
    logits = None
    for i, token in enumerate(curr_tokens):
        # Passing pos=i
        logits = model.forward(token, i, kv_cache)

    # Generate
    for step in range(args.steps):
        next_token = sample_logits(logits, temp=args.temp)

        # Decode and print
        piece = llm.detokenize([next_token]).decode("utf-8", errors="ignore")
        print(piece, end="", flush=True)

        # Stop on EOS
        if next_token == llm.token_eos():
            print("\n<EOS>")
            break

        # Forward next
        curr_tokens.append(next_token)
        logits = model.forward(next_token, len(curr_tokens) - 1, kv_cache)

    print("\n\n--- Done ---")

    if args.ref:
        print("\n--- Reference (llama.cpp) ---")
        # We need a fresh full load for inference if we want to check reference
        llm_ref = llama_cpp.Llama(model_path=args.model, verbose=False)
        output = llm_ref(args.prompt, max_tokens=args.steps, temperature=args.temp)
        print(args.prompt + output["choices"][0]["text"])


if __name__ == "__main__":
    main()
