import random

import numpy as np

from femr.stat_utils import ReservoirSampler


def test_reservoir_sample_is_not_biased_to_the_first_samples():
    random.seed(0)
    sampler = ReservoirSampler(1_000)
    for i in range(100_000):
        sampler.add(i, 1.0)

    samples = np.array(sampler.samples)
    assert len(samples) == 1_000
    # A uniform sample has ~1% of its items among the first 1,000 and a median near 50,000.
    # The old sampler kept >99% of the first 1,000 items.
    assert np.mean(samples < 1_000) < 0.05
    assert 40_000 < np.median(samples) < 60_000


def test_reservoir_respects_weights_and_combine():
    random.seed(0)
    light, heavy = ReservoirSampler(1_000), ReservoirSampler(1_000)
    for _ in range(20_000):
        light.add(0, 1.0)
        heavy.add(1, 3.0)
    light.combine(heavy)
    # Value 1 carries 3/4 of the total weight.
    assert 0.65 < np.mean(light.samples) < 0.85
