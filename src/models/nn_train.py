import time
import numpy as np
import torch
import torch.nn as nn


def _batches(n, bs, gen=None, shuffle=False):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


class NNWrapper:
    def __init__(self, params, task, seed, meta, cfg, name):
        self.p, self.task, self.seed, self.meta = params, task, seed, meta
        self.ncfg = cfg["neural"]
        self.time_cap = self.ncfg["trial_time_cap_s"][name]
        torch.set_num_threads(self.ncfg["torch_threads"])
        self.timed_out = False
        self.epochs_run = 0

    def build(self):
        raise NotImplementedError

    def param_groups(self):
        return self.net.parameters()

    def _raw(self, xn, xc):
        a = xn if xn.shape[1] > 0 else None
        b = xc if xc.shape[1] > 0 else None
        return self.net(a, b).squeeze(-1)

    def _predict_raw(self, xn, xc):
        self.net.eval()
        outs = []
        with torch.no_grad():
            for b in _batches(len(xn), 4096):
                outs.append(self._raw(xn[b], xc[b]))
        return torch.cat(outs)

    def fit(self, X, y, Xva, yva):
        torch.manual_seed(self.seed)
        clf = self.task == "classification"
        xn, xc = (torch.as_tensor(a) for a in X)
        vn, vc = (torch.as_tensor(a) for a in Xva)
        y = np.asarray(y, dtype=np.float32)
        yva = np.asarray(yva, dtype=np.float32)
        if not clf:
            self.ym, self.ys = float(y.mean()), float(y.std() + 1e-8)
            y, yva = (y - self.ym) / self.ys, (yva - self.ym) / self.ys
        yt, yv = torch.as_tensor(y), torch.as_tensor(yva)

        self.build()
        opt = torch.optim.AdamW(self.param_groups(), lr=self.p["lr"],
                                weight_decay=self.p["weight_decay"])
        lossf = nn.BCEWithLogitsLoss() if clf else nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        bs = self.p["batch_size"]
        best, best_state, bad = float("inf"), None, 0
        t0 = time.time()
        for epoch in range(self.ncfg["max_epochs"]):
            self.net.train()
            for b in _batches(len(yt), bs, gen, True):
                if time.time() - t0 > self.time_cap:
                    self.timed_out = True
                    break
                opt.zero_grad()
                loss = lossf(self._raw(xn[b], xc[b]), yt[b])
                loss.backward()
                opt.step()
            v = float(lossf(self._predict_raw(vn, vc), yv))
            if v < best - 1e-6:
                best = v
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
                bad = 0
            else:
                bad += 1
            self.epochs_run = epoch + 1
            if self.timed_out or bad >= self.ncfg["early_stopping_patience"]:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        xn, xc = (torch.as_tensor(a) for a in X)
        raw = self._predict_raw(xn, xc)
        if self.task == "classification":
            return torch.sigmoid(raw).numpy()
        return raw.numpy() * self.ys + self.ym