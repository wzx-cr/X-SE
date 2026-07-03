from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


_CONTROL_KEYS = {
    "name",
    "init",
    "debug",
    "source_dir",
    "root",
    "repo_dir",
    "config",
    "source_config",
    "config_group",
    "variant",
    "size",
    "model_kwargs",
    "sample_rate",
    "mono_mode",
    "return_dict",
    "return_spec",
    "flatten_parameters",
    "remove_weight_reparameterizations",
    "param",
    "strict",
    "state_prefix",
    "forward_type",
    "input_type",
    "spec_data",
    "mag_data",
    "mask",
    "in_channels",
    "mag_decoder",
    "spec_decoder",
    "rnn",
}

_VARIANT_CONFIG_PREFIX = {
    "dprnn": "dprnn",
    "dptransformer": "dpt",
    "ln": "ln",
    "time_kernel": "time_kernel",
}

_DEFAULT_MODEL_KWARGS_BY_SIZE = {
    "t": {
        "channels": 24,
        "kernel_size": [8, 3, 3],
        "stride": 4,
        "rnnformer_kwargs": {
            "num_blocks": 2,
            "channels": 20,
            "freq": 16,
            "num_heads": 4,
            "eps": 1.0e-5,
            "positional_embedding": "train",
            "attn_bias": False,
            "post_act": False,
            "pre_norm": False,
        },
        "pre_post_init": "linear_fixed",
        "n_fft": 512,
        "hop_size": 256,
        "win_size": 512,
        "window": "hann",
        "stft_normalized": False,
        "mask": None,
        "activation": "SiLU",
        "activation_kwargs": {"inplace": True},
        "input_compression": 0.3,
        "normalize_final_conv": True,
        "weight_norm": True,
        "resnet": False,
    },
    "b": {
        "channels": 48,
        "kernel_size": [8, 3, 3],
        "stride": 4,
        "rnnformer_kwargs": {
            "num_blocks": 3,
            "channels": 36,
            "freq": 24,
            "num_heads": 4,
            "eps": 1.0e-5,
            "positional_embedding": "train",
            "attn_bias": False,
            "post_act": False,
            "pre_norm": False,
        },
        "pre_post_init": "linear_fixed",
        "n_fft": 512,
        "hop_size": 256,
        "win_size": 512,
        "window": "hann",
        "stft_normalized": False,
        "mask": None,
        "activation": "SiLU",
        "activation_kwargs": {"inplace": True},
        "input_compression": 0.3,
        "normalize_final_conv": True,
        "weight_norm": True,
        "resnet": False,
    },
    "s": {
        "channels": 64,
        "kernel_size": [8, 3, 3, 3],
        "stride": 4,
        "rnnformer_kwargs": {
            "num_blocks": 3,
            "channels": 48,
            "freq": 36,
            "num_heads": 4,
            "eps": 1.0e-5,
            "positional_embedding": "train",
            "attn_bias": False,
            "post_act": False,
            "pre_norm": False,
        },
        "pre_post_init": "linear_fixed",
        "n_fft": 512,
        "hop_size": 256,
        "win_size": 512,
        "window": "hann",
        "stft_normalized": False,
        "mask": None,
        "activation": "SiLU",
        "activation_kwargs": {"inplace": True},
        "input_compression": 0.3,
        "normalize_final_conv": True,
        "weight_norm": True,
        "resnet": False,
    },
    "m": {
        "channels": 96,
        "kernel_size": [8, 3, 3, 3],
        "stride": 4,
        "rnnformer_kwargs": {
            "num_blocks": 4,
            "channels": 72,
            "freq": 48,
            "num_heads": 4,
            "eps": 1.0e-5,
            "positional_embedding": "train",
            "attn_bias": False,
            "post_act": False,
            "pre_norm": False,
        },
        "pre_post_init": "linear_fixed",
        "n_fft": 512,
        "hop_size": 160,
        "win_size": 512,
        "window": "hann",
        "stft_normalized": False,
        "mask": None,
        "activation": "SiLU",
        "activation_kwargs": {"inplace": True},
        "input_compression": 0.3,
        "normalize_final_conv": True,
        "weight_norm": True,
        "resnet": False,
    },
    "l": {
        "channels": 128,
        "kernel_size": [8, 3, 3, 3, 3],
        "stride": 4,
        "rnnformer_kwargs": {
            "num_blocks": 5,
            "channels": 96,
            "freq": 64,
            "num_heads": 4,
            "eps": 1.0e-5,
            "positional_embedding": "train",
            "attn_bias": False,
            "post_act": False,
            "pre_norm": False,
            "p_dropout": 0.0,
        },
        "pre_post_init": "linear_fixed",
        "n_fft": 512,
        "hop_size": 100,
        "win_size": 512,
        "window": "hann",
        "stft_normalized": False,
        "mask": None,
        "activation": "SiLU",
        "activation_kwargs": {"inplace": True},
        "input_compression": 0.3,
        "normalize_final_conv": True,
        "weight_norm": True,
        "resnet": False,
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _normalize_variant(variant: str | None) -> str:
    if not variant:
        return "default"
    return str(variant).split(".")[-1]


def _resolve_source_dir(conf: Mapping[str, Any]) -> Path | None:
    source_dir = (
        conf.get("source_dir")
        or conf.get("repo_dir")
        or conf.get("root")
    )
    if source_dir is None or str(source_dir).lower() in {"", "internal", "builtin"}:
        return None
    path = Path(str(source_dir)).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    path = path.resolve()
    if not path.exists():
        # Backward compatibility for configs created before FastEnhancer was
        # folded into this framework. A missing legacy `fastenhancer-main`
        # directory means "use the built-in default core".
        if Path(str(source_dir)).name in {"fastenhancer-main", "fastenhancer_main"}:
            return None
        raise FileNotFoundError(f"FastEnhancer source directory not found: {path}")
    return path


def _resolve_config_path(
    source_dir: Path | None,
    conf: Mapping[str, Any],
    variant: str,
) -> Path | None:
    config_path = conf.get("source_config") or conf.get("config")
    if config_path:
        path = Path(str(config_path)).expanduser()
        if not path.is_absolute():
            source_candidate = source_dir / path if source_dir is not None else None
            repo_candidate = _repo_root() / path
            path = source_candidate if source_candidate is not None and source_candidate.exists() else repo_candidate
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"FastEnhancer config not found: {path}")
        return path

    size = conf.get("size")
    if not size or source_dir is None:
        return None

    size = str(size).lower()
    config_group = str(conf.get("config_group", "fastenhancer"))
    if config_group == "ablation":
        prefix = _VARIANT_CONFIG_PREFIX.get(variant)
        if prefix is None:
            raise ValueError(
                f"FastEnhancer variant `{variant}` has no ablation preset. "
                "Set model.config/source_config explicitly."
            )
        file_name = f"{prefix}_{size}.yaml"
    else:
        file_name = f"{size}.yaml"

    path = source_dir / "configs" / config_group / file_name
    if not path.exists():
        raise FileNotFoundError(f"FastEnhancer preset config not found: {path}")
    return path.resolve()


def _load_source_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        size = path.stem.lower()
        group = path.parent.name
        if group in {"fastenhancer", "fastenhancer_dns"} and size in _DEFAULT_MODEL_KWARGS_BY_SIZE:
            return {
                "model": "fastenhancer.default",
                "model_kwargs": deepcopy(_DEFAULT_MODEL_KWARGS_BY_SIZE[size]),
            }
        raise ModuleNotFoundError(
            "PyYAML is required to read custom FastEnhancer config files. "
            "Install `pyyaml` or use model.config_group=fastenhancer with "
            "model.size=t|b|s|m|l."
        )
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _load_model_class(source_dir: Path | None, variant: str):
    variant = _normalize_variant(variant)
    if source_dir is None:
        if variant != "default":
            raise ValueError(
                "The built-in FastEnhancer integration currently includes only "
                "`fastenhancer.default`. Set model.source_dir to an external "
                "FastEnhancer checkout for other variants."
            )
        from modules.blocks.fastenhancer_core import Model
        return Model

    model_path = source_dir / "models" / "fastenhancer" / variant / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"FastEnhancer model file not found: {model_path}")

    source_dir_str = str(source_dir)
    if source_dir_str not in sys.path:
        sys.path.insert(0, source_dir_str)

    functional_module = sys.modules.get("functional")
    if functional_module is not None:
        module_file = getattr(functional_module, "__file__", None)
        expected_dir = (source_dir / "functional").resolve()
        if module_file is not None:
            try:
                Path(module_file).resolve().relative_to(expected_dir)
            except ValueError as exc:
                raise ImportError(
                    "`functional` is already imported from a different package. "
                    f"FastEnhancer expects `{expected_dir}`."
                ) from exc

    module_name = f"_mos_fastenhancer_{variant}_{abs(hash(str(model_path.resolve())))}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load FastEnhancer module from {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    if not hasattr(module, "Model"):
        raise AttributeError(f"FastEnhancer module `{model_path}` does not define `Model`.")
    return module.Model


def resolve_fastenhancer_model_kwargs(
    model_conf: Mapping[str, Any] | None = None,
) -> tuple[Path | None, str, dict[str, Any]]:
    """Resolve official FastEnhancer config plus framework overrides."""

    conf = _to_plain(model_conf or {})
    source_dir = _resolve_source_dir(conf)
    variant = _normalize_variant(conf.get("variant"))
    config_path = _resolve_config_path(source_dir, conf, variant)
    source_config = _load_source_config(config_path)

    source_model = source_config.get("model")
    if source_model:
        variant = _normalize_variant(source_model)

    if "model_kwargs" in source_config:
        model_kwargs = _to_plain(source_config.get("model_kwargs", {}))
    else:
        size = str(conf.get("size", "b")).lower()
        model_kwargs = deepcopy(_DEFAULT_MODEL_KWARGS_BY_SIZE.get(size, {}))

    direct_overrides = {
        key: value for key, value in conf.items() if key not in _CONTROL_KEYS
    }
    if "hop_length" in direct_overrides and "hop_size" not in direct_overrides:
        direct_overrides["hop_size"] = direct_overrides.pop("hop_length")
    if "win_length" in direct_overrides and "win_size" not in direct_overrides:
        direct_overrides["win_size"] = direct_overrides.pop("win_length")

    model_kwargs.update(direct_overrides)
    model_kwargs.update(_to_plain(conf.get("model_kwargs", {})))
    return source_dir, variant, model_kwargs


def create_fastenhancer_model(model_conf: Mapping[str, Any] | None = None):
    conf = _to_plain(model_conf or {})
    source_dir, variant, model_kwargs = resolve_fastenhancer_model_kwargs(conf)
    model_class = _load_model_class(source_dir, variant)
    model = model_class(**model_kwargs)

    if conf.get("flatten_parameters", True) and hasattr(model, "flatten_parameters"):
        model.flatten_parameters()
    return model
