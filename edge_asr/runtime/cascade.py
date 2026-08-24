"""The three-stage cascade that makes the two-model split correct on
SW6100 / Snapdragon Wear Elite.

    [eNPU, always-on, ~mW]   VAD -> command/KWS model (Model 2)
                                       |
                                       +-- direct action (fast path)
                                       +-- "wake" ---------------+
                                                                 v
    [Hexagon, duty-cycled]              streaming ASR (Model 1) --> text

Model 2 stays resident on the ultra-low-power eNPU and gates the expensive
Hexagon path. This class wires the two together in Python so the control
flow is testable; on-device the two stages live on different NPUs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from .streaming_session import StreamingASRSession


@dataclass
class CascadePipeline:
    command_model: object          # BCResNet
    command_classes: List[str]
    asr_session: StreamingASRSession
    frontend: object
    cmvn: object
    wake_class: str = "wake"
    confidence_threshold: float = 0.6
    on_command: Optional[Callable[[str], None]] = None

    awake: bool = False

    def _command_infer(self, wav: torch.Tensor):
        feats = self.frontend(wav).squeeze(0)
        feats = self.cmvn(feats.unsqueeze(0)).squeeze(0)
        logits = self.command_model(feats.unsqueeze(0))
        probs = logits.softmax(-1).squeeze(0)
        idx = int(probs.argmax())
        return self.command_classes[idx], float(probs[idx])

    @torch.no_grad()
    def process(self, wav: torch.Tensor) -> dict:
        """Feed one utterance-sized buffer. Returns what fired."""
        result = {"stage": None, "command": None, "text": None}
        if not self.awake:
            cls, conf = self._command_infer(wav)
            result["stage"] = "command"
            if conf >= self.confidence_threshold and cls != "unknown":
                result["command"] = cls
                if cls == self.wake_class:
                    self.awake = True
                    self.asr_session.reset()
                elif self.on_command:
                    self.on_command(cls)
            return result
        # awake -> stream into the big model
        text = self.asr_session.accept_waveform(wav)
        result["stage"] = "asr"
        result["text"] = text
        if self.asr_session.endpointed:
            self.awake = False  # go back to low-power gating
            result["final"] = True
        return result
