import os
from pathlib import Path
from omegaconf import OmegaConf


def args_parser():
    conf = OmegaConf.create({
        'conf': 'conf/config.yaml',
        'cmd': 'train'
    })
    conf.merge_with_cli()
    conf_path = Path(conf.conf)
    if not conf_path.exists():
        # graceful fallback to a known sample config if default is missing
        fallback = Path(__file__).resolve().parent.parent.parent.joinpath('1.yaml')
        if fallback.exists():
            print(f"[args_parser] {conf_path} not found, fallback to {fallback}")
            conf_path = fallback
        else:
            raise FileNotFoundError(f"Config file not found: {conf.conf}. "
                                    "Pass --conf <your_config.yaml>.")
    conf = OmegaConf.merge(OmegaConf.load(conf_path), conf)
    OmegaConf.resolve(conf)
    return conf
