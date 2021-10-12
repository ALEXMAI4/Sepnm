import pylab
import numpy as np
import json
import pathlib

A = 0


def f(x):
    return 0.5 + (
        np.sin(x ** 2 - A ** 2) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + A ** 2))


x_min = -10
x_max = 10
dx = 0.001

x = np.arange(x_min, x_max, dx)
y = f(x)

res = {
    "x": x.tolist(),
    "y": y.tolist(),
}

path = pathlib.Path("results")
path.mkdir(exist_ok=True)
file = path / "result.json"
# out = open(file, "w")
out = file.open("w")
# file = open("result.json", "w")
# file.write(json.dumps(res, indent=4))
json.dump(res, out, indent=4)
out.close()


pylab.plot(x, y)
pylab.grid()
# pylab.show()
pylab.savefig("results/figure.png")