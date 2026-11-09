"""DiffuSeq as a model family of the attack (see elf_inversion.models)."""
from .adapter import DiffuSeqAdapter as Adapter
from .forward import ForwardMetadata
from .setup import BackboneSetup

OBSERVATION_OPTIONS = ()


def load_metadata(fields: dict) -> ForwardMetadata:
    return ForwardMetadata(**fields)
