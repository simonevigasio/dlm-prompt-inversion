"""What the attack needs from a victim model, on top of Stage B's TokenModel.

The attack itself (backbone/, tokens/, pipeline.py, observation.py) only talks to
a model through these methods; everything model-specific lives in models/<family>/.
"""
from typing import Protocol, runtime_checkable

from ..backbone.interface import ForwardMap
from ..backbone.knowledge import AttackerKnowledge
from ..tokens.interface import TokenModel


@runtime_checkable
class VictimModel(TokenModel, Protocol):
    family: str            # name in models.FAMILIES, saved with every observation
    sampling_steps: int    # sampler calls of the model's default schedule
    provenance: dict       # where the weights come from; the command saves it in manifest.json

    def with_prompt_length(self, prompt_length: int) -> "VictimModel": ...
    # Public metadata of one observation. ``step`` is the observed sampler call (0 = the first)
    # of a sampler run with ``sampling_steps`` calls; the family converts it to its own time
    # value. **options are the family's OBSERVATION_OPTIONS.
    def metadata(self, prompt_length: int, observation_layer: int, *, step: int,
                 sampling_steps: int, **options): ...
    # Stage A's differentiable map: coordinates -> observed hidden-state rows.
    def forward_map(self, metadata, knowledge: AttackerKnowledge = AttackerKnowledge()) -> ForwardMap: ...
    # Run the model's own sampler and capture the hidden state (an observation.SyntheticExample):
    # the prompt rows, then the generated (noise) rows, and no other row.
    # This is the oracle the forward map is checked against, never part of the inversion.
    def observe(self, tokens, metadata, *, noise_seed: int, sampling_steps: int): ...
