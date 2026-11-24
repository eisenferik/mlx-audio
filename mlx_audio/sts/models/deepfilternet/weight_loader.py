"""
Weight loading utilities for DeepFilterNet models.

Handles conversion between PyTorch and MLX parameter naming conventions.
"""

import re
from typing import Dict, List, Set

import mlx.core as mx
from mlx.utils import tree_flatten


def _apply_gru_mapping(name: str) -> str:
    """Map PyTorch GRU parameter names to MLX GRU layer names."""
    gru_comp_map = {
        "weight_ih": "Wx",
        "weight_hh": "Wh",
        "bias_ih": "b",
        "bias_hh": "bhn",
    }

    m = re.search(r"\.gru\.(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)$", name)
    if not m:
        return name

    comp = gru_comp_map[m.group(1)]
    layer = m.group(2)
    return re.sub(
        r"\.gru\.(weight_ih|weight_hh|bias_ih|bias_hh)_l\d+$",
        f".gru_layers.{layer}.{comp}",
        name,
    )


def _apply_stride_conv_index_offset(name: str) -> str:
    """Map PyTorch Sequential indices to dict-based conv block indices."""
    stride_conv_patterns = [
        r"^enc\.erb_conv[123]\.",
        r"^enc\.df_conv1\.",
    ]

    if not any(re.match(pattern, name) for pattern in stride_conv_patterns):
        return name

    parts = name.split(".")
    if len(parts) >= 3 and parts[2].isdigit():
        parts[2] = str(int(parts[2]) + 1)
        return ".".join(parts)
    return name


def _sequential_layer_candidates(name: str) -> List[str]:
    """Return candidate names where PyTorch Sequential idx is represented as .layers.idx in MLX."""
    candidates = [name]
    replacements = {
        ".linear_in.0.": ".linear_in.layers.0.",
        ".linear_in.1.": ".linear_in.layers.1.",
        ".linear_out.0.": ".linear_out.layers.0.",
        ".linear_out.1.": ".linear_out.layers.1.",
        ".df_fc_emb.0.": ".df_fc_emb.layers.0.",
        ".df_fc_emb.1.": ".df_fc_emb.layers.1.",
        ".lsnr_fc.0.": ".lsnr_fc.layers.0.",
        ".lsnr_fc.1.": ".lsnr_fc.layers.1.",
        ".df_fc_a.0.": ".df_fc_a.layers.0.",
        ".df_fc_a.1.": ".df_fc_a.layers.1.",
        ".df_out.0.": ".df_out.layers.0.",
        ".df_out.1.": ".df_out.layers.1.",
        ".clc_fc_a.0.": ".clc_fc_a.layers.0.",
        ".clc_fc_a.1.": ".clc_fc_a.layers.1.",
        ".clc_fc_out.0.": ".clc_fc_out.layers.0.",
        ".clc_fc_out.1.": ".clc_fc_out.layers.1.",
        ".fc_emb.0.": ".fc_emb.layers.0.",
    }

    for src, dst in replacements.items():
        if src in name:
            candidates.append(name.replace(src, dst))
    return candidates


def get_weight_mapping(pt_names: Set[str], mlx_names: Set[str]) -> Dict[str, str]:
    """Build mapping from PyTorch weight names to MLX names.

    Returns:
        Dictionary mapping PyTorch names to MLX names
    """
    mapping = {}
    has_model_prefix = any(name.startswith("model.") for name in mlx_names)

    for pt_name in pt_names:
        if "num_batches_tracked" in pt_name or pt_name.endswith(".h0"):
            continue

        base = _apply_stride_conv_index_offset(_apply_gru_mapping(pt_name))
        # DF1 conv blocks expose pointwise conv as `pwconv`.
        base = base.replace(".1x1conv.", ".pwconv.")
        candidates: List[str] = []

        # Keep raw pt_name as a fallback because some names are already aligned.
        candidates.append(base)
        candidates.extend(_sequential_layer_candidates(base))
        if pt_name != base:
            candidates.append(pt_name)
            candidates.extend(_sequential_layer_candidates(pt_name))

        if has_model_prefix:
            candidates.extend([f"model.{c}" for c in list(candidates)])

        # Deduplicate while preserving first-match behavior.
        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            if cand in mlx_names:
                mapping[pt_name] = cand
                break

    return mapping


def _resolve(model, name: str):
    """Return (parent, attr) for a dot-separated path, or (None, None)."""
    parts = name.split(".")
    obj = model

    try:
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            elif part.isdigit():
                idx = int(part)
                if hasattr(obj, "__getitem__") and len(obj) > idx:
                    obj = obj[idx]
                else:
                    return None, None
            elif hasattr(obj, "__getitem__"):
                try:
                    obj = obj[part]
                except (KeyError, IndexError, TypeError):
                    return None, None
            else:
                return None, None
        return obj, parts[-1]
    except (KeyError, AttributeError, TypeError, IndexError):
        return None, None


def _get_weight(model, name: str):
    obj, attr = _resolve(model, name)
    if obj is None or attr is None:
        return None
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict) and attr in obj:
        return obj[attr]
    return None


def _folded_gru_biases(bias_ih: mx.array, bias_hh: mx.array, bhn_current: mx.array):
    """Map PyTorch GRU biases onto MLX nn.GRU layout.

    MLX nn.GRU stores bhn as the n-gate only [H], while PyTorch bias_hh is [3H].
    r/z of bias_hh are added to bias_ih. Shapes that already match are unchanged.
    """
    if (
        hasattr(bhn_current, "shape")
        and len(bhn_current.shape) == 1
        and len(bias_hh.shape) == 1
        and bhn_current.shape[0] * 3 == bias_hh.shape[0]
    ):
        h = bhn_current.shape[0]
        rz = bias_hh[: 2 * h]
        n_gate = bias_hh[2 * h :]
        rz_pad = mx.concatenate([rz, mx.zeros((h,), dtype=bias_ih.dtype)], axis=0)
        return bias_ih + rz_pad, n_gate
    return bias_ih, bias_hh


def _load_gru_bias_pair(model, pair: Dict[str, tuple]) -> int:
    """Apply bias_ih and bias_hh together so the fold cannot depend on arrival order."""
    b_entry = pair.get("b")
    bhn_entry = pair.get("bhn")
    if b_entry is not None and bhn_entry is not None:
        mlx_b, bias_ih = b_entry
        mlx_bhn, bias_hh = bhn_entry
        current_bhn = _get_weight(model, mlx_bhn)
        if current_bhn is not None:
            bias_ih, bias_hh = _folded_gru_biases(bias_ih, bias_hh, current_bhn)
        loaded = 0
        if set_weight(model, mlx_bhn, bias_hh):
            loaded += 1
        if set_weight(model, mlx_b, bias_ih):
            loaded += 1
        return loaded

    loaded = 0
    for entry in (b_entry, bhn_entry):
        if entry is None:
            continue
        mlx_name, value = entry
        if set_weight(model, mlx_name, value):
            loaded += 1
    return loaded


def set_weight(model, name: str, value: mx.array) -> bool:
    """Set a weight in the model by dot-separated path.

    Handles dict-based modules with string keys and integer indices for lists.

    Args:
        model: The model object
        name: Dot-separated path (e.g., 'enc.erb_conv0.1.weight')
        value: The weight value

    Returns:
        True if successful, False otherwise
    """
    obj, final_part = _resolve(model, name)
    if obj is None or final_part is None:
        return False

    try:
        if hasattr(obj, final_part):
            # Unpaired PyTorch bias_hh is [3H] while MLX nn.GRU bhn is [H].
            # Keep the n-gate; r/z are folded in _load_gru_bias_pair when both arrive.
            if final_part == "bhn":
                current = getattr(obj, final_part)
                if (
                    hasattr(current, "shape")
                    and len(current.shape) == 1
                    and len(value.shape) == 1
                    and current.shape[0] * 3 == value.shape[0]
                ):
                    h = current.shape[0]
                    setattr(obj, final_part, value[2 * h :])
                    return True
            setattr(obj, final_part, value)
            return True
        elif isinstance(obj, dict) and final_part in obj:
            obj[final_part] = value
            return True
        elif final_part.isdigit():
            idx = int(final_part)
            if hasattr(obj, "__setitem__"):
                obj[idx] = value
                return True
    except (KeyError, AttributeError, TypeError, IndexError):
        pass

    return False


def load_weights(model, weights: Dict[str, mx.array]) -> int:
    """Load PyTorch weights into MLX model.

    Args:
        model: MLX DfNet model
        weights: Dictionary of weights from PyTorch checkpoint

    Returns:
        Number of weights loaded
    """
    # Get model parameter names
    param_tree = tree_flatten(model.parameters())
    mlx_names = {k for k, v in param_tree}
    pt_names = set(weights.keys())

    # Build mapping
    mapping = get_weight_mapping(pt_names, mlx_names)

    # GRU b / bhn must be applied as a pair. Mapping iteration follows a set of
    # strings, whose order is process-randomized by PYTHONHASHSEED.
    pairs: Dict[str, Dict[str, tuple]] = {}
    rest = []
    for pt_name, mlx_name in mapping.items():
        attr = mlx_name.rsplit(".", 1)[-1]
        if attr in ("b", "bhn"):
            parent = mlx_name.rsplit(".", 1)[0]
            pairs.setdefault(parent, {})[attr] = (mlx_name, weights[pt_name])
        else:
            rest.append((mlx_name, weights[pt_name]))

    loaded = 0
    for mlx_name, value in rest:
        if set_weight(model, mlx_name, value):
            loaded += 1

    for pair in pairs.values():
        loaded += _load_gru_bias_pair(model, pair)

    # Load filterbanks - handle both wrapper and direct model cases
    if "erb_fb" in weights:
        if hasattr(model, "model") and hasattr(model.model, "erb_fb"):
            model.model.erb_fb = weights["erb_fb"]
        elif hasattr(model, "erb_fb"):
            model.erb_fb = weights["erb_fb"]

    if "mask.erb_inv_fb" in weights:
        if hasattr(model, "model"):
            # DeepFilterNet wrapper
            if hasattr(model.model, "erb_inv_fb"):
                model.model.erb_inv_fb = weights["mask.erb_inv_fb"]
            if hasattr(model.model, "mask"):
                model.model.mask.erb_inv_fb = weights["mask.erb_inv_fb"]
        elif hasattr(model, "mask"):
            # Direct DfNet
            model.mask.erb_inv_fb = weights["mask.erb_inv_fb"]

    return loaded
