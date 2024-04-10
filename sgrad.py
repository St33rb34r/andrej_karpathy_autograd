import math


class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, value):
        value = value if isinstance(value, Value) else Value(value)
        out = Value(self.data + value.data, (self, value), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            value.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, value):
        value = value if isinstance(value, Value) else Value(value)
        out = Value(self.data * value.data, (self, value), '*')

        def _backward():
            self.grad += value.data * out.grad
            value.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __sub__(self, value):
        value = value if isinstance(value, Value) else Value(value)
        out = Value(self.data - value.data, (self, value), '-')

        def _backward():
            self.grad += 1.0 * out.grad
            value.grad += -1.0 * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, value):
        value = value if isinstance(value, Value) else Value(value)
        out = Value(self.data / value.data, (self, value), '/')

        def _backward():
            self.grad += (out.grad / value.data)
            value.grad += (-self.data / (value.data * value.data)) * (out.grad)

        out._backward = _backward

        return out

    def __pow__(self, scalar):
        assert isinstance(scalar, (int, float))
        scalar = scalar if isinstance(scalar, Value) else Value(scalar)
        out = Value(self.data ** scalar, (self,), f'**{scalar}')

        def _backward():
            self.grad += scalar * out.grad * (self.data ** (scalar - 1))

        out._backward = _backward

    def __rmul__(self, value):
        return self * value

    def __radd__(self, value):
        return self + value

    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - (t * t)) * out.grad

        out._backward = _backward

        return out

    def relu(self):

        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')

        def _backward():
            self.grad += 1.0 * out.grad if self.data > 0 else 0

        out._backward = _backward

        return out

    def backward(self):
        topolist = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topolist.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topolist):
            node._backward()

    def ackward(self, earlier_derivative=None):

        if len(self._prev) == 1:
            child1 = list(self._prev)[0]

            if self._op == 'tanh':
                n = child1.data
                t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
                child1.grad += (1 - (t * t))

            if self._op == 'relu':
                if self.data > 0:
                    self.grad = 1
                else:
                    self.grad = 0

            if earlier_derivative:
                child1.grad *= earlier_derivative
            else:
                self.grad = 1

            if len(child1._prev) > 0:
                child1.ackward(child1.grad)

        elif len(self._prev) == 2:
            child1, child2 = self._prev

            if self._op == '+':
                child1.grad += 1
                child2.grad += 1

            elif self._op == '*':
                child1.grad += child2.data
                child2.grad += child1.data

            elif self._op == '-':
                child1.grad += 1
                child2.grad += -1

            elif self._op == '/':
                child2.grad += 1 / child1.data
                child1.grad += - child2.data / (child1.data * child1.data)

            # elif self._op == '**':
            #    child1._grad

            if earlier_derivative:
                child1.grad *= earlier_derivative
                child2.grad *= earlier_derivative
            else:
                self.grad = 1

            if len(child1._prev) > 0:
                child1.ackward(child1.grad)

            if len(child2._prev) > 0:
                child2.ackward(child2.grad)

    def zero_grad(self):

        if len(self._prev) > 0:
            child1, child2 = self._prev
            child1.grad, child2.grad = 0, 0

            if len(child1._prev) > 0:
                child1.zero_grad()

            if len(child2._prev) > 0:
                child2.zero_grad()