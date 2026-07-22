"""Non-mutating signal capture for Qwen3.5 Gated DeltaNet calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any

import torch

from .quantization import QuantizationSpec, quantize_dequantize


@dataclass(frozen=True, slots=True)
class GatedDeltaSignal:
    call_index: int
    layer_index: int
    sequence_length: int
    had_initial_state: bool
    beta_mean: float
    beta_min: float
    beta_max: float
    retention_mean: float
    retention_min: float
    retention_max: float
    initial_state_l2: float | None
    final_state_l2: float | None
    state_update_relative_l2: float | None
    committed_residual_rms: float | None
    probe_state_relative_l2: float | None
    probe_read_error_rms: float | None
    probe_read_relative_l2: float | None

    def evidence_dict(self) -> dict[str, object]:
        return asdict(self)


def _argument(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    name: str,
    position: int,
) -> object | None:
    if name in kwargs:
        return kwargs[name]
    return args[position] if len(args) > position else None


def _committed_residual_rms(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> float | None:
    if key.shape[1] != 1:
        return None
    key_float = key[:, 0].to(torch.float32)
    key_float = key_float * torch.rsqrt(
        key_float.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
    )
    value_float = value[:, 0].to(torch.float32)
    retention = g[:, 0].to(torch.float32).exp().unsqueeze(-1).unsqueeze(-1)
    decayed_state = initial_state.to(torch.float32) * retention
    previous_value = (decayed_state * key_float.unsqueeze(-1)).sum(dim=-2)
    residual = (value_float - previous_value) * beta[:, 0].to(torch.float32).unsqueeze(-1)
    return float(residual.square().mean().sqrt().item())


def _probe_state_read_error(
    *,
    query: torch.Tensor,
    initial_state: torch.Tensor,
    spec: QuantizationSpec,
) -> tuple[float, float, float] | None:
    if query.shape[1] != 1:
        return None
    query_float = query[:, 0].to(torch.float32)
    query_float = query_float * torch.rsqrt(
        query_float.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
    )
    state_float = initial_state.to(torch.float32)
    qdq = quantize_dequantize(state_float, spec)
    state_error = qdq.tensor.to(torch.float32) - state_float
    reference_read = (state_float * query_float.unsqueeze(-1)).sum(dim=-2)
    read_error = (state_error * query_float.unsqueeze(-1)).sum(dim=-2)
    read_error_norm = torch.linalg.vector_norm(read_error)
    reference_norm = torch.linalg.vector_norm(reference_read).clamp_min(1e-12)
    return (
        qdq.relative_l2_error,
        float(read_error.square().mean().sqrt().item()),
        float((read_error_norm / reference_norm).item()),
    )


class GatedDeltaSignalRecorder:
    """Wrap GDN function pointers and record signals without changing outputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        probe_spec: QuantizationSpec | None = None,
    ) -> None:
        self.model = model
        self.probe_spec = probe_spec
        self.records: list[GatedDeltaSignal] = []
        self.enabled = True
        self._installed: list[tuple[object, str, object]] = []

    def _gdn_modules(self) -> list[torch.nn.Module]:
        modules = [
            module
            for module in self.model.modules()
            if module.__class__.__name__ == "Qwen3_5GatedDeltaNet"
        ]
        if not modules:
            raise TypeError("model does not contain Qwen3_5GatedDeltaNet modules")
        return modules

    def _make_wrapper(self, module: torch.nn.Module, original: Any):
        layer_index = int(module.layer_idx)

        def wrapped(*args: object, **kwargs: object):
            query = _argument(args, kwargs, "query", 0)
            key = _argument(args, kwargs, "key", 1)
            value = _argument(args, kwargs, "value", 2)
            g = _argument(args, kwargs, "g", 3)
            beta = _argument(args, kwargs, "beta", 4)
            initial_state = _argument(args, kwargs, "initial_state", 5)
            required = (query, key, value, g, beta)
            if self.enabled and not all(isinstance(item, torch.Tensor) for item in required):
                raise TypeError("Gated DeltaNet call did not expose tensor q/k/v/g/beta inputs")

            initial_snapshot = None
            residual_rms = None
            probe = None
            if self.enabled and isinstance(initial_state, torch.Tensor):
                initial_snapshot = initial_state.detach().clone()
                residual_rms = _committed_residual_rms(
                    key=key,
                    value=value,
                    g=g,
                    beta=beta,
                    initial_state=initial_snapshot,
                )
                if self.probe_spec is not None:
                    probe = _probe_state_read_error(
                        query=query,
                        initial_state=initial_snapshot,
                        spec=self.probe_spec,
                    )

            output = original(*args, **kwargs)
            if not self.enabled:
                return output

            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("Gated DeltaNet kernel must return (output, final_state)")
            final_state = output[1]
            beta_float = beta.detach().to(torch.float32)
            retention = g.detach().to(torch.float32).exp()
            initial_l2 = None
            final_l2 = None
            update_ratio = None
            if initial_snapshot is not None and isinstance(final_state, torch.Tensor):
                initial_float = initial_snapshot.to(torch.float32)
                final_float = final_state.detach().to(torch.float32)
                initial_norm = torch.linalg.vector_norm(initial_float)
                final_norm = torch.linalg.vector_norm(final_float)
                update_norm = torch.linalg.vector_norm(final_float - initial_float)
                initial_l2 = float(initial_norm.item())
                final_l2 = float(final_norm.item())
                update_ratio = float((update_norm / initial_norm.clamp_min(1e-12)).item())

            self.records.append(
                GatedDeltaSignal(
                    call_index=len(self.records),
                    layer_index=layer_index,
                    sequence_length=int(query.shape[1]),
                    had_initial_state=initial_snapshot is not None,
                    beta_mean=float(beta_float.mean().item()),
                    beta_min=float(beta_float.min().item()),
                    beta_max=float(beta_float.max().item()),
                    retention_mean=float(retention.mean().item()),
                    retention_min=float(retention.min().item()),
                    retention_max=float(retention.max().item()),
                    initial_state_l2=initial_l2,
                    final_state_l2=final_l2,
                    state_update_relative_l2=update_ratio,
                    committed_residual_rms=residual_rms,
                    probe_state_relative_l2=probe[0] if probe is not None else None,
                    probe_read_error_rms=probe[1] if probe is not None else None,
                    probe_read_relative_l2=probe[2] if probe is not None else None,
                )
            )
            return output

        return wrapped

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("recorder is already installed")
        for module in self._gdn_modules():
            for attribute in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
                original = getattr(module, attribute)
                setattr(module, attribute, self._make_wrapper(module, original))
                self._installed.append((module, attribute, original))

    def remove(self) -> None:
        while self._installed:
            module, attribute, original = self._installed.pop()
            setattr(module, attribute, original)

    def __enter__(self) -> GatedDeltaSignalRecorder:
        self.install()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.remove()
