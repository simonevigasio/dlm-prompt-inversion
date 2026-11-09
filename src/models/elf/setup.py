"""Victim-model setup for running experiments on different ELF backbones"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BackboneSetup:
    # weights: "checkpoint" loads the released ELF denoiser, "random" draws fresh
    # parameters from init_seed. bottleneck=False makes the text projection square
    # (bottleneck_dim = T5 d_model) and is only possible with random weights.
    # init: "upstream" is the initialization ELF was trained from (only with the
    # checkpoint); "normal" draws every parameter from N(0, init_std^2).
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
            raise ValueError("The checkpoint weights were trained using bottleneck, then if you want to avoid the bottleneck self.weights needs to be set to \"random\"")
        if self.weights == "checkpoint" and self.init != "upstream":
            raise ValueError("The checkpoint was trained using the upstream initialization, so when self.weights is set to \"checkpoint\", then self.init needs to be set to \"upstream\" for consistency")
        if self.weights == "random" and self.init == "upstream":
            raise ValueError("Random weights with the upstream initialization are not an experiment we run (the final layer is zero-initialized), so when self.weights is set to \"random\", self.init cannot be \"upstream\"")
