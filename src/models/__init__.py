"""Victim models the attack can invert, by family name.

Each family lives in ``models/<name>/`` and has a ``family.py`` module with:

    Adapter              the VictimModel class (see interface.py), with from_config(path, setup=...)
    ForwardMetadata      the public metadata its forward map declares
    BackboneSetup        the victim-weights toggles of a pipeline config's ``backbone:`` section
    OBSERVATION_OPTIONS  model-specific keys of ``synthetic_observation`` (passed to Adapter.metadata)
    load_metadata(dict)  ForwardMetadata from a saved observation.json

A model settings file (e.g. configs/model.yaml) names its family with ``family:``;
files without the key are ELF settings.
"""
from importlib import import_module
from pathlib import Path

import yaml

FAMILIES = ("elf", "diffuseq")
DEFAULT_FAMILY = "elf"


def family(name: str):
    """The ``family.py`` module of a model family."""
    if name not in FAMILIES:
        raise ValueError(f"Unknown model family {name!r}; known: {list(FAMILIES)}")
    return import_module(f"elf_inversion.models.{name}.family")


def settings_family(path) -> str:
    """Family named by a model settings file."""
    settings = yaml.safe_load(Path(path).read_text()) or {}
    return settings.get("family", DEFAULT_FAMILY)
