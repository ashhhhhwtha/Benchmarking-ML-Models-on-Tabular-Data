import numpy as np


def get_space(spaces, model, task):
    s = spaces[model]
    return s.get(task, s)


def sample_params(space, rng):
    out = {}
    for k, s in space.items():
        t = s["type"]
        if t == "loguniform":
            out[k] = float(np.exp(rng.uniform(np.log(s["low"]), np.log(s["high"]))))
        elif t == "uniform":
            out[k] = float(rng.uniform(s["low"], s["high"]))
        elif t == "int":
            out[k] = int(rng.randint(s["low"], s["high"] + 1))
        elif t == "choice":
            out[k] = s["values"][rng.randint(len(s["values"]))]
        else:
            raise ValueError(f"unknown type {t} for {k}")
    return out