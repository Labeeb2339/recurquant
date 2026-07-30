"""Causal StateLease observation for Qwen3.5 recurrent-state caches.

Qwen3.5 constructs its Gated DeltaNet transition tensors immediately before
calling either the chunk or recurrent state kernel.  This context manager
observes those exact inputs, the kernel's initial state, and its returned final
state.  It stages one observation only after the selected kernel succeeds and
before the model writes the final state to the selected cache.

Dispatch is by cache identity.  Reference, static, and otherwise unregistered
caches are never inspected or modified.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch

_OBSERVER_ATTRIBUTE = "_recurquant_statelease_observer"
_PACKED_SEQUENCE_ARGUMENTS = (
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "cu_seqlens",
)
_CACHE_TRANSACTION_METHODS = (
    "begin_statelease_forward_transaction",
    "commit_statelease_forward_transaction",
    "rollback_statelease_forward_transaction",
)
_KERNEL_ARGUMENT_POSITIONS = {
    "chunk_gated_delta_rule": {
        "initial_state": 6,
        "use_qk_l2norm_in_kernel": 8,
    },
    "recurrent_gated_delta_rule": {
        "initial_state": 5,
        "use_qk_l2norm_in_kernel": 7,
    },
}


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


class Qwen35StateLeaseObserver:
    """Stage exact Qwen3.5 Gated DeltaNet transitions on StateLease caches.

    ``caches`` must implement
    ``stage_statelease_observation(layer_idx, query, key, value, log_decay,
    beta, initial_state, final_state)`` and
    ``discard_pending_statelease_observation(layer_idx)``, an explicit pending
    receipt, and the begin/commit/rollback model-forward transaction contract.
    Only those exact objects are observed.

    The observer supports uncached prefill, cached multi-token chunks, and
    cached one-token decode.  It can be nested with RecurQuant's query-energy
    and transition observers in either context-manager order.  Packed
    ``cu_seqlens`` forwards are rejected because StateLease replay requires one
    unambiguous chronological transition sequence.
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
            raise TypeError("caches must be a list or tuple of StateLease caches")
        if not caches:
            raise ValueError("caches must contain at least one StateLease cache")

        identities = [id(cache) for cache in caches]
        if len(set(identities)) != len(identities):
            raise ValueError("caches must not contain the same object more than once")
        for cache in caches:
            if not callable(getattr(cache, "stage_statelease_observation", None)):
                raise TypeError("each observed cache must implement stage_statelease_observation")
            if not callable(getattr(cache, "discard_pending_statelease_observation", None)):
                raise TypeError(
                    "each observed cache must implement discard_pending_statelease_observation"
                )
            if not callable(getattr(cache, "has_pending_statelease_observation", None)):
                raise TypeError(
                    "each observed cache must implement has_pending_statelease_observation"
                )
            for method_name in _CACHE_TRANSACTION_METHODS:
                if not callable(getattr(cache, method_name, None)):
                    raise TypeError(f"each observed cache must implement {method_name}")

        self.model = model
        self.caches = tuple(caches)
        self._cache_by_identity = {id(cache): cache for cache in self.caches}
        self._installed: list[tuple[object, str, object]] = []
        self._active_calls: ContextVar[tuple[_ForwardCall, ...]] = ContextVar(
            f"recurquant_statelease_calls_{id(self)}",
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
        cache.discard_pending_statelease_observation(  # type: ignore[attr-defined]
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
                "RecurQuant could not discard the pending StateLease observation "
                f"after the forward failed: {cleanup_error!r}"
            )

    def _make_kernel_wrapper(
        self,
        module: torch.nn.Module,
        original: Any,
        kernel_name: str,
    ):
        positions = _KERNEL_ARGUMENT_POSITIONS[kernel_name]

        def wrapped(*args: object, **kwargs: object):
            active_calls = self._active_calls.get()
            call = active_calls[-1] if active_calls else None
            if call is None or call.module is not module:
                return original(*args, **kwargs)
            if call.kernel_calls:
                raise RuntimeError(
                    "Qwen3.5 Gated DeltaNet called more than one state kernel in one "
                    "forward; refusing ambiguous StateLease attribution"
                )

            # A failed torch or FLA kernel must never leave a staged transition.
            output = original(*args, **kwargs)
            call.kernel_calls += 1

            observed = {
                "query": _argument(args, kwargs, "query", 0),
                "key": _argument(args, kwargs, "key", 1),
                "value": _argument(args, kwargs, "value", 2),
                "log_decay": _argument(args, kwargs, "g", 3),
                "beta": _argument(args, kwargs, "beta", 4),
                "initial_state": _argument(
                    args,
                    kwargs,
                    "initial_state",
                    positions["initial_state"],
                ),
            }
            use_qk_l2norm = _argument(
                args,
                kwargs,
                "use_qk_l2norm_in_kernel",
                positions["use_qk_l2norm_in_kernel"],
            )
            if use_qk_l2norm is not True:
                raise RuntimeError(
                    "StateLease requires the pinned Qwen3.5 kernel to consume "
                    "L2-normalized query/key tensors"
                )
            malformed = [
                name
                for name, value in observed.items()
                if name != "initial_state" and not isinstance(value, torch.Tensor)
            ]
            initial_state = observed["initial_state"]
            if initial_state is not None and not isinstance(initial_state, torch.Tensor):
                malformed.append("initial_state")
            if malformed:
                rendered = ", ".join(malformed)
                raise TypeError(
                    "Qwen3.5 Gated DeltaNet kernel did not expose valid StateLease "
                    f"inputs ({rendered})"
                )
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError(
                    "Qwen3.5 Gated DeltaNet kernel did not return exactly (output, final_state)"
                )
            final_state = output[1]
            if not isinstance(final_state, torch.Tensor):
                raise TypeError("Qwen3.5 Gated DeltaNet kernel did not return a tensor final state")

            call.stage_attempted = True
            call.cache.stage_statelease_observation(  # type: ignore[attr-defined]
                call.layer_index,
                observed["query"],
                observed["key"],
                observed["value"],
                observed["log_decay"],
                observed["beta"],
                initial_state,
                final_state,
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
                name for name in _PACKED_SEQUENCE_ARGUMENTS if kwargs.get(name) is not None
            ]
            if packed_arguments:
                rendered = ", ".join(packed_arguments)
                raise ValueError(
                    "Qwen3.5 StateLease observation does not support packed "
                    f"cu_seqlens forwards ({rendered})"
                )

            call = _ForwardCall(module=module, cache=cache, layer_index=layer_index)
            token = self._active_calls.set((*self._active_calls.get(), call))
            try:
                output = original(*args, **kwargs)
                if call.kernel_calls != 1 or not call.stage_attempted or not call.staged:
                    raise RuntimeError(
                        "Qwen3.5 Gated DeltaNet did not complete exactly one observed "
                        "state-kernel call; refusing ambiguous StateLease inputs"
                    )
                if call.cache.has_pending_statelease_observation(  # type: ignore[attr-defined]
                    call.layer_index
                ):
                    raise RuntimeError(
                        "Qwen3.5 Gated DeltaNet returned before its staged "
                        "StateLease transition was consumed by the cache"
                    )
                return output
            except BaseException as error:
                self._discard_after_failure(call, error)
                raise
            finally:
                self._active_calls.reset(token)

        return wrapped

    def _selected_forward_caches(
        self,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[object, ...]:
        selected: list[object] = []
        seen: set[int] = set()
        for candidate in (*args, *kwargs.values()):
            cache = self._selected_cache(candidate)
            if cache is None or id(cache) in seen:
                continue
            selected.append(cache)
            seen.add(id(cache))
        return tuple(selected)

    def _make_model_forward_wrapper(self, original: Any):
        def wrapped(*args: object, **kwargs: object):
            transactions: list[tuple[object, object]] = []
            try:
                for cache in self._selected_forward_caches(args, kwargs):
                    transaction = cache.begin_statelease_forward_transaction()  # type: ignore[attr-defined]
                    transactions.append((cache, transaction))
                output = original(*args, **kwargs)
                for cache, transaction in transactions:
                    cache.commit_statelease_forward_transaction(  # type: ignore[attr-defined]
                        transaction
                    )
                return output
            except BaseException as error:
                for cache, transaction in reversed(transactions):
                    try:
                        cache.rollback_statelease_forward_transaction(  # type: ignore[attr-defined]
                            transaction
                        )
                    except BaseException as cleanup_error:
                        error.add_note(
                            "RecurQuant could not roll back the complete cache after "
                            f"the model forward failed: {cleanup_error!r}"
                        )
                raise

        return wrapped

    def install(self) -> None:
        """Install forward and selected torch/FLA kernel wrappers atomically."""

        if self._installed:
            raise RuntimeError("StateLease observer is already installed")
        modules = self._gdn_modules()
        observed_objects: list[object] = [self.model, *modules]
        if any(hasattr(item, _OBSERVER_ATTRIBUTE) for item in observed_objects):
            raise RuntimeError("a StateLease observer is already installed on this model")

        try:
            setattr(self.model, _OBSERVER_ATTRIBUTE, self)
            self._installed.append((self.model, _OBSERVER_ATTRIBUTE, _Missing))
            original_model_forward = self.model.forward
            if not callable(original_model_forward):
                raise TypeError("model.forward must be callable")
            model_restore_value = (
                original_model_forward if "forward" in self.model.__dict__ else _Missing
            )
            self.model.forward = self._make_model_forward_wrapper(  # type: ignore[method-assign]
                original_model_forward
            )
            self._installed.append((self.model, "forward", model_restore_value))
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
                        raise TypeError(f"Qwen3_5GatedDeltaNet.{attribute} must be callable")
                    wrapper = (
                        self._make_forward_wrapper(module, original)
                        if attribute == "forward"
                        else self._make_kernel_wrapper(module, original, attribute)
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

    def __enter__(self) -> Qwen35StateLeaseObserver:
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
