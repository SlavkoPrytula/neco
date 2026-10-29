"""Offset-addressable data sampling and worker seeding."""
import random
import numpy as np
import torch
from torch.utils.data import Sampler


class FieldSampler(Sampler):
    """Deterministic shuffled passes; resume by consumed fields, not prefetched fields."""
    def __init__(self, size, seed, offset, count):
        if size <= 0 or offset < 0 or count < 0:
            raise ValueError('Invalid sample stream bounds')
        self.size, self.seed, self.offset, self.count = size, seed, offset, count

    def __len__(self):
        return self.count

    def __iter__(self):
        cycle, position = divmod(self.offset, self.size)
        remaining = self.count
        while remaining:
            order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + cycle)).tolist()
            take = min(remaining, self.size - position)
            yield from order[position:position + take]
            remaining -= take
            cycle, position = cycle + 1, 0


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
