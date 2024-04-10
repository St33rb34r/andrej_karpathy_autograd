import random
from typing import List

from .sgrad import Value


class Neuron:

    def __init__(self, num_in: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_in)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out


class Layer:

    def __init__(self, num_in: int, num_out: int):
        self.num_in = num_in
        self.num_out = num_out

        self.neurons = [Neuron(num_in) for _ in range(num_out)]

    def __call__(self, x):
        out = [[n(xi) for n in self.neurons] for xi in x]
        return out


class MLP:

    def __init__(self, layout: List[int]):
        self.layers = [Layer(layout[i], layout[i + 1]) for i in range(len(layout) - 1)]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x