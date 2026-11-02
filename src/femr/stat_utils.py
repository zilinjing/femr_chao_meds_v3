import dataclasses
import math
import random


@dataclasses.dataclass
class OnlineStatistics:
    """
    A class for computing online statistics such as mean and variance.
    From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
    Welford's online algorithm: we don't need to store all the values, we can just store the mean and variance and update them as we go.
    """

    count: float
    current_mean: float
    variance: float

    def __init__(self):
        """
        Initialize online statistics. Optionally takes the results of self.to_dict() to initialize from old data.
        """
        self.count = 0
        self.current_mean = 0
        self.variance = 0

    def add(self, weight: float, value: float) -> None:
        """
        Add an observation to the calculation.
        for a new value with weight, we update the count,mean and variance as follows:
        
        """
        self.count += weight
        delta = value - self.current_mean
        self.current_mean += delta * (weight / self.count)
        delta2 = value - self.current_mean

        self.variance += weight * (delta * delta2)

    def mean(self) -> float:
        """
        Return the current mean.
        """
        return self.current_mean

    def standard_deviation(self) -> float:
        """
        Return the current standard devation.
        """
        return math.sqrt(self.variance / self.count)

    def combine(self, other) -> None:
        if self.count == 0:
            self.count = other.count
            self.current_mean = other.current_mean
            self.variance = other.variance
        elif other.count != 0:
            total = self.count + other.count
            delta = other.current_mean - self.current_mean
            new_mean = self.current_mean + delta * (other.count / total)
            new_variance = self.variance + other.variance + (delta * self.count) * (delta * other.count) / total

            self.count = total
            self.current_mean = new_mean
            self.variance = new_variance


class ReservoirSampler:
    def __init__(self, size):
        self.total_weight = 0
        self.size = size
        self.samples = []

    def add(self, sample, weight):
        # Weighted reservoir sampling (Chao, 1982): once full, the new sample enters with
        # probability size * weight / total_weight and replaces a uniformly chosen slot.
        # (The previous version used weight / total_weight, which kept almost only the first samples.)
        self.total_weight += weight
        if len(self.samples) < self.size:
            self.samples.append(sample)
        elif random.random() < self.size * weight / self.total_weight:
            self.samples[random.randint(0, self.size - 1)] = sample

    def combine(self, other):
        if not other.samples:
            return
        if len(self.samples) == self.size and len(other.samples) == other.size:
            # Both full: each slot comes from `other` with probability W_other / (W_self + W_other).
            p_other = other.total_weight / (self.total_weight + other.total_weight)
            pool = list(other.samples)
            random.shuffle(pool)
            for i in range(self.size):
                if pool and random.random() < p_other:
                    self.samples[i] = pool.pop()
            self.total_weight += other.total_weight
            return
        if len(self.samples) < self.size and len(other.samples) == other.size:
            # Start from the full reservoir and replay the smaller one into it.
            mine, my_weight = self.samples, self.total_weight
            self.samples, self.total_weight = list(other.samples), other.total_weight
            for val in mine:
                self.add(val, my_weight / len(mine))
            return
        for val in other.samples:
            self.add(val, other.total_weight / len(other.samples))
