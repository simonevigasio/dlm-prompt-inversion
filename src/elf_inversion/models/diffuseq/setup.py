"""Victim-model setup for running experiments on different DiffuSeq backbones.

Same toggles and rules as ELF's BackboneSetup. DiffuSeq's bottleneck is its token
embedding width (hidden_dim 128), which ``input_up_proj`` lifts to the 768-wide BERT
encoder. bottleneck=False makes the embeddings as wide as the encoder; upstream then
has no ``input_up_proj`` (the embeddings enter the encoder directly). There is no
checkpoint without the bottleneck, so it needs random weights.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BackboneSetup:
    # weights: "checkpoint" loads the released DiffuSeq model, "random" draws fresh parameters
    # from init_seed. init: "upstream" is the initialization DiffuSeq was trained from (only
    # with the checkpoint); "normal" draws every parameter from N(0, init_std^2).
    weights: Literal["checkpoint", "random"] = "checkpoint"
    bottleneck: bool = True
    init: Literal["upstream", "normal"] = "upstream"
    init_std: float = 0.02
    init_seed: int = 0

    def __post_init__(self):
        if self.weights not in ("checkpoint", "random"):
            raise ValueError("weights must be set to \"checkpoint\" or \"random\"")
        if self.init not in ("upstream", "normal"):
            raise ValueError("init must be set to \"upstream\" or \"normal\"")
        if type(self.bottleneck) is not bool:
            raise ValueError("bottleneck must be boolean")
        if type(self.init_std) not in (int, float) or self.init_std <= 0:
            raise ValueError("init_std must be a positive number")
        if type(self.init_seed) is not int:
            raise ValueError("init_seed must be int")
        if self.weights == "checkpoint" and not self.bottleneck:
            raise ValueError("The checkpoint was trained with the 128-wide embeddings: bottleneck False "
                             "needs weights \"random\"")
        if self.weights == "checkpoint" and self.init != "upstream":
            raise ValueError("The checkpoint was trained from the upstream initialization: weights \"checkpoint\" "
                             "needs init \"upstream\"")
        if self.weights == "random" and self.init == "upstream":
            raise ValueError("Random weights with the upstream initialization are not an experiment we run, "
                             "so weights \"random\" needs init \"normal\"")
