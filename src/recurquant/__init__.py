"""RecurQuant recurrent-state quantization primitives."""

from .evidence import verify_evidence_artifact
from .packed_cache import PackedRecurrentStateCache
from .quantization import (
    PackedQuantizedTensor,
    QuantizationResult,
    QuantizationSpec,
    quantize_dequantize,
    quantize_pack,
)
from .qwen35 import create_qwen35_packed_cache, create_qwen35_v02_mixed_cache

__all__ = [
    "PackedQuantizedTensor",
    "PackedRecurrentStateCache",
    "QuantizationResult",
    "QuantizationSpec",
    "create_qwen35_packed_cache",
    "create_qwen35_v02_mixed_cache",
    "quantize_dequantize",
    "quantize_pack",
    "verify_evidence_artifact",
]

__version__ = "0.2.0a1"
