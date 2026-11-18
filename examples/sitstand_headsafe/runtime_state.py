"""Vectorized deployment transforms, copied from the project alignment implementation; no simulator imports, all values in SI units.

Episode reset is a new warm sensor stream. Controller.reset on an already
running robot must call TargetState.reset only, leaving other states intact.
"""
import torch


def _ids(ids):
    return slice(None) if ids is None else ids


class TargetState:
    def __init__(self, n, home, lower, upper, device='cpu'):
        self.home = torch.as_tensor(home, dtype=torch.float64, device=device)
        self.lower = torch.as_tensor(lower, dtype=torch.float64, device=device)
        self.upper = torch.as_tensor(upper, dtype=torch.float64, device=device)
        self.alpha = torch.tensor([.7]*5+[.5]*4+[.7]*5, dtype=torch.float64, device=device)
        self.previous_unclamped = torch.zeros(n, 14, dtype=torch.float64, device=device)
        self.valid = torch.zeros(n, 1, dtype=torch.bool, device=device)

    def step(self, raw, voltage_multiplier):
        target = self.home + raw.to(self.home.dtype) * voltage_multiplier.reshape(-1, 1).to(self.home.dtype)
        filtered = self.alpha*target + (1-self.alpha)*self.previous_unclamped
        target = torch.where(self.valid, filtered, target)
        self.previous_unclamped.copy_(target)
        self.valid.fill_(True)
        return target.clamp(self.lower, self.upper)

    def reset(self, ids=None):
        ids = _ids(ids)
        self.previous_unclamped[ids] = 0
        self.valid[ids] = False


class VoltageState:
    def __init__(self, n, device='cpu'):
        self.ema = torch.zeros(n, dtype=torch.float64, device=device)

    @property
    def multiplier(self):
        return torch.where(self.ema > 0, 7.4/self.ema.clamp(6.,9.5), 1.)

    def update(self, sample, valid_mask=None):
        sample = sample.reshape(-1).to(self.ema.dtype)
        valid = torch.isfinite(sample) & (sample >= 0)
        if valid_mask is not None:
            valid &= valid_mask
        new = torch.where(self.ema > 0, .1*sample + .9*self.ema, sample)
        self.ema.copy_(torch.where(valid,new,self.ema))

    def reset(self, ids=None):
        self.ema[_ids(ids)] = 0
