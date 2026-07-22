"""Causal transition observation for Qwen3.5 recurrent-state caches.

Qwen3.5 constructs and repeats its Gated DeltaNet query and key heads before
calling either its chunk or recurrent state kernel.  This context manager
observes those exact kernel inputs together with the log decay and write gate.
It stages them only after the kernel succeeds, immediately before the model
writes the resulting recurrent state to the selected cache.

Dispatch is by cache identity.  Reference, static, and otherwise unregistered
caches are never inspected or modified.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch

_OBSERVER_ATTRIBUTE = "_recurquant_transition_observer"
_PACKED_SEQUENCE_ARGUMENTS = (
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "cu_seqlens",
)


def _argument(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    name: str,
    position: int,
) -> object | None:
    if name in kwargs:
        return kwargs[name]
    return args[position] if len(args) > position else None


@dataclass(slots=True)
class _ForwardCall:
    module: torch.nn.Module
    cache: object
    layer_index: int
    kernel_calls: int = 0
    stage_attempted: bool = False
    staged: bool = False


class Qwen35TransitionObserver:
    """Stage exact Qwen3.5 Gated DeltaNet transition inputs on CORA caches.

    ``caches`` must implement
    ``stage_transition_observation(layer_idx, query, key, log_decay, beta)``
    and ``discard_pending_transition_observations(layer_idx)``.  Only those
    exact objects are observed.  The observer supports uncached prefill,
    cached multi-token chunks, and cached one-token decode, and composes with
    :class:`~recurquant.query_energy.Qwen35QueryEnergyObserver` in either
    context-manager nesting order.

    Packed ``cu_seqlens`` forwards are rejected because the frozen CORA-C2
    recurrence requires an unambiguous chronological batch-one sequence.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        caches: list[object] | tuple[object, ...],
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(caches, (list, tuple)):
            raise TypeError("caches must be a list or tuple of CORA caches")
        if not caches:
            raise ValueError("caches must contain at least one CORA cache")

        identities = [id(cache) for cache in caches]
        if len(set(identities)) != len(identities):
            raise ValueError("caches must not contain the same object more than once")
        for cache in caches:
            if not callable(getattr(cache, "stage_transition_observation", None)):
                raise TypeError(
                    "each observed cache must implement stage_transition_observation"
                )
            if not callable(
                getattr(cache, "discard_pending_transition_observations", None)
            ):
                raise TypeError(
                    "each observed cache must implement "
                    "discard_pending_transition_observations"
                )

        self.model = model
        self.caches = tuple(caches)
        self._cache_by_identity = {id(cache): cache for cache in self.caches}
        self._installed: list[tuple[object, str, object]] = []
        self._active_calls: ContextVar[tuple[_ForwardCall, ...]] = ContextVar(
            f"recurquant_transition_calls_{id(self)}",
            default=(),
        )

    def _gdn_modules(self) -> list[torch.nn.Module]:
        modules = [
            module
            for module in self.model.modules()
            if module.__class__.__name__ == "Qwen3_5GatedDeltaNet"
        ]
        if not modules:
            raise TypeError("model does not contain Qwen3_5GatedDeltaNet modules")
        return modules

    def _selected_cache(self, candidate: object | None) -> object | None:
        if candidate is None:
            return None
        selected = self._cache_by_identity.get(id(candidate))
        return selected if selected is candidate else None

    @staticmethod
    def _discard(cache: object, layer_index: int) -> None:
        cache.discard_pending_transition_observations(  # type: ignore[attr-defined]
            layer_index
        )

    def _discard_after_failure(
        self,
        call: _ForwardCall,
        error: BaseException,
    ) -> None:
        try:
            self._discard(call.cache, call.layer_index)
        except BaseException as cleanup_error:
            error.add_note(
                "RecurQuant could not discard the pending transition observation "
                f"after the forward failed: {cleanup_error!r}"
            )

    def _make_kernel_wrapper(
        self,
        module: torch.nn.Module,
        original: Any,
    ):
        def wrapped(*args: object, **kwargs: object):
            active_calls = self._active_calls.get()
            call = active_calls[-1] if active_calls else None
            if call is None or call.module is not module:
                return original(*args, **kwargs)
            if call.kernel_calls:
                raise RuntimeError(
                    "Qwen3.5 Gated DeltaNet called more than one state kernel in one "
                    "forward; refusing ambiguous transition attribution"
                )

            # The selected torch or FLA kernel must complete before its inputs
            # are eligible to be staged.  The Qwen forward writes the returned
            # state on the very next cache operation.
            output = original(*args, **kwargs)
            call.kernel_calls += 1

            observed = {
                "query": _argument(args, kwargs, "query", 0),
                "key": _argument(args, kwargs, "key", 1),
                "log_decay": _argument(args, kwargs, "g", 3),
                "beta": _argument(args, kwargs, "beta", 4),
            }
            malformed = [
                name for name, value in observed.items() if not isinstance(value, torch.Tensor)
            ]
            if malformed:
                rendered = ", ".join(malformed)
                raise TypeError(
                    "Qwen3.5 Gated DeltaNet kernel did not expose tensor transition "
                    f"inputs ({rendered})"
                )

            call.stage_attempted = True
            call.cache.stage_transition_observation(  # type: ignore[attr-defined]
                call.layer_index,
                observed["query"],
                observed["key"],
                observed["log_decay"],
                observed["beta"],
            )
            call.staged = True
            return output

        return wrapped

    def _make_forward_wrapper(
        self,
        module: torch.nn.Module,
        original: Any,
    ):
        layer_index = int(module.layer_idx)  # type: ignore[attr-defined]

        def wrapped(*args: object, **kwargs: object):
            cache = self._selected_cache(_argument(args, kwargs, "cache_params", 1))
            if cache is None:
                return original(*args, **kwargs)

            packed_arguments = [
                name
                for name in _PACKED_SEQUENCE_ARGUMENTS
                if kwargs.get(name) is not None
            ]
            if packed_arguments:
                rendered = ", ".join(packed_arguments)
                raise ValueError(
                    "Qwen3.5 transition observation does not support packed "
                    f"cu_seqlens forwards ({rendered})"
                )

            call = _ForwardCall(module=module, cache=cache, layer_index=layer_index)
            token = self._active_calls.set((*self._active_calls.get(), call))
            try:
                output = original(*args, **kwargs)
                if call.kernel_calls != 1 or not call.stage_attempted or not call.staged:
                    raise RuntimeError(
                        "Qwen3.5 Gated DeltaNet did not complete exactly one observed "
                        "state-kernel call; refusing to continue with ambiguous "
                        "transition inputs"
                    )
                return output
            except BaseException as error:
                self._discard_after_failure(call, error)
                raise
            finally:
                self._active_calls.reset(token)

        return wrapped

    def install(self) -> None:
        """Install forward and selected torch/FLA kernel wrappers atomically."""

        if self._installed:
            raise RuntimeError("transition observer is already installed")
        modules = self._gdn_modules()
        observed_objects: list[object] = [self.model, *modules]
        if any(hasattr(item, _OBSERVER_ATTRIBUTE) for item in observed_objects):
            raise RuntimeError("a transition observer is already installed on this model")

        try:
            setattr(self.model, _OBSERVER_ATTRIBUTE, self)
            self._installed.append((self.model, _OBSERVER_ATTRIBUTE, _Missing))
            for module in modules:
                setattr(module, _OBSERVER_ATTRIBUTE, self)
                self._installed.append((module, _OBSERVER_ATTRIBUTE, _Missing))
                for attribute in (
                    "forward",
                    "chunk_gated_delta_rule",
                    "recurrent_gated_delta_rule",
                ):
                    original = getattr(module, attribute)
                    if not callable(original):
                        raise TypeError(
                            f"Qwen3_5GatedDeltaNet.{attribute} must be callable"
                        )
                    wrapper = (
                        self._make_forward_wrapper(module, original)
                        if attribute == "forward"
                        else self._make_kernel_wrapper(module, original)
                    )
                    restore_value = original if attribute in module.__dict__ else _Missing
                    setattr(module, attribute, wrapper)
                    self._installed.append((module, attribute, restore_value))
        except BaseException:
            self.remove()
            raise

    def remove(self) -> None:
        """Restore every wrapped callable and remove installation markers."""

        while self._installed:
            target, attribute, original = self._installed.pop()
            if original is _Missing:
                if attribute == _OBSERVER_ATTRIBUTE:
                    if getattr(target, attribute, None) is self:
                        delattr(target, attribute)
                elif attribute in getattr(target, "__dict__", {}):
                    delattr(target, attribute)
            else:
                setattr(target, attribute, original)

    def __enter__(self) -> Qwen35TransitionObserver:
        self.install()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.remove()


class _MissingType:
    pass


_Missing = _MissingType()
