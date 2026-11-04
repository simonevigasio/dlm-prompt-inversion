"""Attacker knowledge for running different setup experiments for stage A"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AttackerKnowledge: 
    # In the observed rows, we consider only the language rows, therefore 
    # the time, mode, and scale tokens are not consedered. When we set "all", 
    # we mean prompt + generated rows. When we set "generated", we discard the prompt rows.
    observed_rows: Literal["all", "generated"] = "all"
    knows_time: bool = True
    knows_noise_scale: bool = True

    def __post_init__(self):
        if self.observed_rows not in ("all", "generated"):
            raise ValueError("observed_rows must be set to \"all\" or \"generated\"")
        if type(self.knows_time) is not bool: 
            raise ValueError("knows_time must be boolean")
        if type(self.knows_noise_scale) is not bool: 
            raise ValueError("knows_noise_scale must be boolean")
