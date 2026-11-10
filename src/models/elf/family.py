"""ELF as a model family of the attack (see elf_inversion.models)."""
from .adapter import ELFAdapter as Adapter, load_tokenizer
from .forward import ForwardMetadata
from .setup import BackboneSetup

OBSERVATION_OPTIONS = ("self_cond_cfg_scale", "attention_mask")


def load_metadata(fields: dict) -> ForwardMetadata:
    fields = dict(fields)
    if fields.get("attention_mask") is not None:
        fields["attention_mask"] = tuple(fields["attention_mask"])
    return ForwardMetadata(**fields)
