"""Empty-stack allocations in ActivationCache must follow the cache, not model.cfg.device.

Regression for the layer-0 paths of stack_head_results / stack_neuron_results (and the projected
LN helper), which allocated ``torch.zeros(..., device=model.cfg.device)`` with no dtype. Uses a
stub model whose ``cfg.device`` is deliberately a different device string from the cache tensors;
``meta`` stands in for a GPU the CI box may not have.
"""

import warnings
from types import SimpleNamespace

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache

D_MODEL = 8
POS = 3
N_LAYERS = 2


def _stub_model(cfg_device: str = "meta") -> SimpleNamespace:
    """Minimal model: only what the empty-stack paths touch, with a lying cfg.device."""
    blocks = [
        SimpleNamespace(
            attn=SimpleNamespace(W_O=torch.zeros(2, 4, D_MODEL)),
            mlp=SimpleNamespace(W_out=torch.zeros(6, D_MODEL)),
        )
        for _ in range(N_LAYERS)
    ]
    cfg = SimpleNamespace(
        device=cfg_device,
        n_layers=N_LAYERS,
        n_heads=2,
        d_head=4,
        d_mlp=6,
        d_model=D_MODEL,
        normalization_type="LN",
    )
    return SimpleNamespace(cfg=cfg, blocks=blocks)


def _cache(dtype: torch.dtype = torch.float32, **kwargs) -> ActivationCache:
    cache_dict = {
        "hook_embed": torch.randn(1, POS, D_MODEL, dtype=dtype),
        "blocks.0.hook_resid_post": torch.randn(1, POS, D_MODEL, dtype=dtype),
        "blocks.0.ln1.hook_scale": torch.rand(1, POS, 1, dtype=dtype) + 0.5,
        "ln_final.hook_scale": torch.rand(1, POS, 1, dtype=dtype) + 0.5,
    }
    # compute_head_results (called unconditionally by stack_head_results) reads hook_z per layer.
    for l in range(N_LAYERS):
        cache_dict[f"blocks.{l}.attn.hook_z"] = torch.randn(1, POS, 2, 4, dtype=dtype)
    return ActivationCache(cache_dict, _stub_model(), **kwargs)


# --- stack_head_results, layer 0 -----------------------------------------------------------------


def test_head_results_layer0_follows_cache_device_not_cfg():
    cache = _cache(device="cpu")
    out = cache.stack_head_results(layer=0)
    assert out.shape == (0, 1, POS, D_MODEL)
    assert out.device.type == "cpu"  # cfg.device says "meta"


def test_head_results_layer0_inherits_cache_dtype():
    cache = _cache(dtype=torch.bfloat16, device="cpu")
    out = cache.stack_head_results(layer=0)
    assert out.dtype == torch.bfloat16


def test_head_results_call_device_overrides_constructor():
    cache = _cache(device="meta")
    out = cache.stack_head_results(layer=0, device="cpu")
    assert out.device.type == "cpu"


# --- stack_neuron_results, layer 0 ---------------------------------------------------------------


def test_neuron_results_layer0_follows_cache_device_not_cfg():
    cache = _cache(device="cpu")
    out = cache.stack_neuron_results(layer=0)
    assert out.shape == (0, 1, POS, D_MODEL)
    assert out.device.type == "cpu"


def test_neuron_results_layer0_projected_keeps_projection_dtype():
    cache = _cache(dtype=torch.bfloat16, device="cpu")
    direction = torch.randn(D_MODEL, dtype=torch.bfloat16)
    out = cache.stack_neuron_results(layer=0, project_output_onto=direction)
    assert out.shape == (0, 1, POS)
    assert out.dtype == torch.bfloat16


def test_neuron_results_layer0_projected_with_ln_uses_helper_path():
    cache = _cache(device="cpu")
    direction = torch.randn(D_MODEL, 2)
    out = cache.stack_neuron_results(layer=0, project_output_onto=direction, apply_ln=True)
    assert out.shape == (0, 1, POS, 2)
    assert out.device.type == "cpu"


# --- the `device` argument on every allocation site ----------------------------------------------


@pytest.mark.parametrize("site", ["head", "neuron", "neuron_projected_ln"])
def test_call_device_overrides_constructor_device(site):
    """Call-level device wins at all three sites; the constructor device is 'meta' to prove it."""
    cache = _cache(device="meta")
    direction = torch.randn(D_MODEL, 2)
    if site == "head":
        out = cache.stack_head_results(layer=0, device="cpu")
    elif site == "neuron":
        out = cache.stack_neuron_results(layer=0, device="cpu")
    else:
        out = cache.stack_neuron_results(
            layer=0, project_output_onto=direction, apply_ln=True, device="cpu"
        )
    assert out.shape[0] == 0
    assert out.device.type == "cpu"


@pytest.mark.parametrize("site", ["head", "neuron", "neuron_projected_ln"])
def test_constructor_device_used_when_call_omits_it(site):
    cache = _cache(device=torch.device("cpu"))
    direction = torch.randn(D_MODEL, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no inference → no FutureWarning
        if site == "head":
            out = cache.stack_head_results(layer=0)
        elif site == "neuron":
            out = cache.stack_neuron_results(layer=0)
        else:
            out = cache.stack_neuron_results(layer=0, project_output_onto=direction, apply_ln=True)
    assert out.device == torch.device("cpu")


def test_device_argument_accepts_str_and_torch_device():
    cache = _cache(device="meta")
    assert cache.stack_head_results(layer=0, device="cpu").device.type == "cpu"
    assert cache.stack_head_results(layer=0, device=torch.device("cpu")).device.type == "cpu"


def test_call_device_does_not_mutate_cache_device():
    cache = _cache(device="meta")
    cache.stack_head_results(layer=0, device="cpu")
    assert cache.device == torch.device("meta")


def test_empty_stack_concats_with_cache_tensors():
    """The point of the fix: the placeholder must be cat-able with real cache-derived stacks."""
    cache = _cache(dtype=torch.bfloat16, device="cpu")
    empty = cache.stack_head_results(layer=0)
    resid = cache["blocks.0.hook_resid_post"][None]
    stacked = torch.cat([empty, resid], dim=0)
    assert stacked.shape == (1, 1, POS, D_MODEL)
    assert stacked.dtype == torch.bfloat16


# --- device resolution order and the inference FutureWarning -------------------------------------


def test_no_device_anywhere_infers_from_cache_with_one_future_warning():
    cache = _cache()  # no constructor device
    with pytest.warns(FutureWarning, match="inferred device"):
        out = cache.stack_head_results(layer=0)
    assert out.device.type == "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # second call on the same cache must be silent
        cache.stack_neuron_results(layer=0)


def test_constructor_device_silences_inference_warning():
    cache = _cache(device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cache.stack_head_results(layer=0)


def test_to_records_device_and_silences_inference_warning():
    cache = _cache().to("cpu")
    assert cache.device == torch.device("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cache.stack_head_results(layer=0)


def test_cfg_device_is_never_consulted():
    """cfg.device is an unresolvable string; touching it would raise inside torch."""
    cache = _cache(device="cpu")
    cache.model.cfg.device = "not-a-device"
    cache.stack_head_results(layer=0)
    cache.stack_neuron_results(layer=0)


# --- repr and __getitem__ ------------------------------------------------------------------------


def test_repr_short_cache_lists_all_keys():
    cache = ActivationCache({"hook_embed": torch.zeros(1)}, _stub_model())
    assert repr(cache) == "ActivationCache with keys ['hook_embed']"


def test_repr_long_cache_is_truncated_with_count():
    cache_dict = {f"blocks.{i}.hook_resid_post": torch.zeros(1) for i in range(40)}
    text = repr(ActivationCache(cache_dict, _stub_model()))
    assert text.startswith("ActivationCache with 40 keys [")
    assert "'blocks.0.hook_resid_post'" in text
    assert "'blocks.39.hook_resid_post'" not in text
    assert "+32 more" in text
    assert len(text) < 400


def test_getitem_accepts_full_name_shorthand_and_tuples():
    cache = _cache(device="cpu")
    full = cache["blocks.0.hook_resid_post"]
    assert cache[("resid_post", 0)] is full
    assert cache[("resid_post", -N_LAYERS)] is full  # negative layer index
    assert cache[("z", 0, "attn")] is cache["blocks.0.attn.hook_z"]
    assert cache["embed"] is cache["hook_embed"]
