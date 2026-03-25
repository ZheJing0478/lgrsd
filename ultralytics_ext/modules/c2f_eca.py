from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f

from .eca import ECA


class C2fECA(C2f):
    """C2f block with an ECA attention applied on the block output.

    This design keeps the parent C2f parameter names intact, so pretrained YOLOv8 weights can be loaded
    with minimal mismatch (only the added ECA parameters are missing).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        k_size: int = 3,
    ) -> None:
        super().__init__(c1=c1, c2=c2, n=n, shortcut=shortcut, g=g, e=e)
        self.eca = ECA(k_size=k_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return self.eca(y)


