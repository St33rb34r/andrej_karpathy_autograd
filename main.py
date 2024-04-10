from nn import MLP


def mse(pred, gt):
    cost = sum((p - y)*(p - y) for p, y in zip(pred, gt))
    return cost


m = MLP([2, 3, 5, 3, 1])
x = [[1, 2], [1, 2], [1, 4], [347, 3]]
y = [-1, -1, 1, 1]

lr = 0.01
params = m.parameters()

losses = []
preds = []

for _ in range(10):

    m.zero_grad()

    ypred = m(x)

    loss = mse(ypred, y)

    loss.backward()

    preds.append([xi.data for xi in ypred])
    losses.append(loss.data)

    for p in m.parameters():
        p.data -= lr * p.grad
