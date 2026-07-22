"""RecurQuant recurrent-state quantization primitives."""

from .packed_cache import PackedRecurrentStateCache
from .quantization import (
    PackedQuantizedTensor,
    QuantizationResult,
    QuantizationSpec,
    quantize_dequantize,
    quantize_pack,
)
from .qwen35 import create_qwen35_packed_cache

__all__ = [
    "PackedQuantizedTensor",
    "PackedRecurrentStateCache",
    "QuantizationResult",
    "QuantizationSpec",
    "create_qwen35_packed_cache",
    "quantize_dequantize",
    "quantize_pack",
]

__version__ = "0.2.0.dev0"
