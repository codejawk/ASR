"""Flash-paged expert store — the mechanism behind the novelty.

Classical sparse MoE is *backwards* for a watch: it saves compute (which the
Hexagon has plenty of) but must keep **every expert resident in RAM** (the
scarce resource). We invert that: experts live in **flash**, and only a
small **resident cache** (typically 1 expert) is ever in RAM. A router
selects the expert per utterance; on a miss we page it in from flash,
measuring the load latency — which hides behind the wake-word gate.

Result: large *effective* capacity (sum of all experts, on cheap flash)
under a tiny *resident* footprint (one expert in RAM).

`ExpertPager` is storage-agnostic: `expert_paths` map expert-id -> a file on
disk (our stand-in for the flash partition), and `builder()` constructs a
fresh module to load weights into. Resident cache is LRU.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class PagingStats:
    hits: int = 0
    misses: int = 0
    total_load_s: float = 0.0

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def as_dict(self):
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hit_rate, 3),
                "avg_load_ms": round(1000 * self.total_load_s / max(self.misses, 1), 2)}


class ExpertPager:
    def __init__(self, expert_paths: Dict[int, str], builder: Callable[[], nn.Module],
                 resident_capacity: int = 1, device: str = "cpu"):
        self.paths = expert_paths
        self.builder = builder
        self.capacity = resident_capacity
        self.device = device
        self._resident: "OrderedDict[int, nn.Module]" = OrderedDict()
        self.stats = PagingStats()

    def flash_footprint_mb(self) -> float:
        return sum(os.path.getsize(p) for p in self.paths.values()) / 1e6

    def resident_footprint_mb(self) -> float:
        total = 0
        for m in self._resident.values():
            total += sum(p.numel() * p.element_size() for p in m.parameters())
        return total / 1e6

    def get(self, expert_id: int) -> Tuple[nn.Module, float]:
        """Return (expert_module, load_latency_seconds). 0 latency on a hit."""
        if expert_id in self._resident:
            self._resident.move_to_end(expert_id)
            self.stats.hits += 1
            return self._resident[expert_id], 0.0

        # miss -> page in from flash
        self.stats.misses += 1
        t0 = time.perf_counter()
        module = self.builder().to(self.device)
        blob = torch.load(self.paths[expert_id], map_location=self.device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        module.load_state_dict(state)
        module.eval()
        latency = time.perf_counter() - t0
        self.stats.total_load_s += latency

        # evict LRU if over capacity
        self._resident[expert_id] = module
        self._resident.move_to_end(expert_id)
        while len(self._resident) > self.capacity:
            self._resident.popitem(last=False)
        return module, latency

    def report(self) -> dict:
        return {
            "experts": len(self.paths),
            "resident_capacity": self.capacity,
            "flash_footprint_mb": round(self.flash_footprint_mb(), 2),
            "resident_footprint_mb": round(self.resident_footprint_mb(), 2),
            **self.stats.as_dict(),
        }
