"""Experimental token-loop guard; a stop marks incomplete output, never refusal.

Exact repetition is a mechanical quality flag, not a semantic coherence classifier.
Callers must exclude loop-stopped responses from final behavioral denominators (or
record an error), preserve raw output, and avoid labeling them comply/refuse.
"""
from __future__ import annotations

import math

import torch


class LoopStop:
    """HF-compatible batch-one stopping callable with sparse CPU suffix checks.

    Inspect generated tokens only, at multiples of 16 generated tokens once
    ``min_tokens`` have been generated. A matching suffix must contain an exact
    period repeated at least ``min_repeats`` times and span at least ``min_tokens``
    tokens. Defaults inspect at most 512 tail tokens per checkpoint. The configured
    window grows only if needed for larger caller-supplied limits.
    """
    checkpoint_interval = 16

    def __init__(self, prompt_length, min_tokens=64, max_period=128, min_repeats=4):
        for name,value,minimum in [('prompt_length',prompt_length,0),('min_tokens',min_tokens,1),
                                   ('max_period',max_period,1),('min_repeats',min_repeats,2)]:
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        self.prompt_length, self.min_tokens = prompt_length,min_tokens
        self.max_period,self.min_repeats = max_period,min_repeats
        self.window = max(min_tokens,max_period*min_repeats)
        self.triggered = False
        self.generated_tokens = 0
        self.period = None
        self.repeat_span = 0
        self._last_check = None

    def __call__(self,input_ids,scores=None,**kwargs):
        if (not isinstance(input_ids,torch.Tensor) or input_ids.ndim != 2 or
                input_ids.shape[0] != 1 or input_ids.shape[1] < self.prompt_length):
            raise ValueError('LoopStop requires batch-one IDs including the full prompt')
        if self.triggered:
            return True
        generated = input_ids.shape[1]-self.prompt_length
        self.generated_tokens = generated
        if (generated < self.min_tokens or generated % self.checkpoint_interval or
                generated == self._last_check):
            return False
        self._last_check = generated
        # One bounded device-to-host transfer per checkpoint, not per-token scans.
        count = min(generated,self.window)
        tail = input_ids[0,-count:].detach().cpu().tolist()
        for period in range(1,min(self.max_period,len(tail)//self.min_repeats)+1):
            minimum_repeats = max(self.min_repeats,math.ceil(self.min_tokens/period))
            if period*minimum_repeats > len(tail):
                continue
            unit = tail[-period:]
            span = period*minimum_repeats
            if tail[-span:] != unit*minimum_repeats:
                continue
            # Record the maximal observed exact suffix span, bounded by the tail.
            repeats = minimum_repeats
            while period*(repeats+1) <= len(tail):
                left = len(tail)-period*(repeats+1)
                if tail[left:left+period] != unit:
                    break
                repeats += 1
            self.triggered,self.period,self.repeat_span = True,period,period*repeats
            return True
        return False

    def report(self):
        return dict(triggered=self.triggered,generated_tokens=self.generated_tokens,
                    period=self.period,repeat_span=self.repeat_span,
                    reason='repeated_token_loop' if self.triggered else None,
                    incomplete=self.triggered)
