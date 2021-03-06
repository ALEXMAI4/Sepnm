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

x = np.arange(x_min, x_max+dx, dx)
y = f(x)

res = {
    "x": x.tolist(),
    "y": y.tolist(),
}

path = pathlib.Path("results")
path.mkdir(exist_ok=True)
file = path / "result_task1.json"
# 2 способ out = open(file, "w")
out = file.open("w")
# 2 способ file = open("result.json", "w")
# 2 способ file.write(json.dumps(res, indent=4))
json.dump(res, out)
out.close()


pylab.plot(x, y)
pylab.grid()
pylab.savefig("results/task1.png")
pylab.show()
