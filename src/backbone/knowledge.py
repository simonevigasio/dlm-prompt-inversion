"""What the attacker is assumed to know; each flag switches one Stage A experiment."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class AttackerKnowledge:
    # An observation only ever holds prompt rows and generated (noise) rows: never a model's
    # prefix or format rows (ELF's time/scale/mode tokens, DiffuSeq's [CLS]/[SEP]).
    # "all": prompt + generated rows. "generated": the prompt rows are hidden.
    observed_rows: Literal["all", "generated"] = "all"
    # False: Stage A recovers the sampler time as one more unknown.
    knows_time: bool = True
    # True: Stage A's random starts draw the generated rows with the model's own noise scale.
    # False: they use noise_scale_guess instead (the model's scale is not looked at).
    knows_noise_scale: bool = True
    noise_scale_guess: float = 1.0

    def __post_init__(self):
        if self.observed_rows not in ("all", "generated"):
            raise ValueError("observed_rows must be set to \"all\" or \"generated\"")
        if type(self.knows_time) is not bool:
            raise ValueError("knows_time must be boolean")
        if type(self.knows_noise_scale) is not bool:
            raise ValueError("knows_noise_scale must be boolean")
        guess = self.noise_scale_guess
        if type(guess) not in (int, float) or not np.isfinite(guess) or guess < 0:
            raise ValueError("noise_scale_guess must be a finite, nonnegative number")

    def noise_scale(self, model_scale: float) -> float:
        """Std of the starting noise: the model's scale if known, else the guess."""
        return model_scale if self.knows_noise_scale else self.noise_scale_guess
