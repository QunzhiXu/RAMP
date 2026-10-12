"""Self-supervised masked-marker pretraining, then a frozen linear probe.

The question is not "which arm trains a better small classifier" — the main
grid answers that — but whether an arm's alignment lets cells from two
platforms be *merged into one pretraining corpus*.  So the comparison holds the
cell budget fixed and varies only the composition of the corpus: 2N cells from
one platform against N + N from two, exactly the A@2N / A+B@2N contrast the
main grid uses for M4, and evaluated on a third platform held out of both.

The pretraining objective is masked-marker reconstruction, which is why the
downstream probe is cell type and rare-population detection and never Job 2:
Job 2 *is* the pretraining objective, and scoring it here would measure nothing
but how well the encoder learned its own task.
"""
import numpy as np, torch, torch.nn as nn

MASK_RATE = 0.15
LATENT = 128
HIDDEN = (128, 64)
EPOCHS = 400
BATCH = 512
LR = 1e-3
PATIENCE = 15
VAL_FRAC = 0.10


class MaskedAutoencoder(nn.Module):
    """Input is the masked values concatenated with the mask indicator, so the
    encoder can tell a masked entry from a genuine zero."""

    def __init__(self, k):
        super().__init__()
        h1, h2 = HIDDEN
        self.enc = nn.Sequential(nn.Linear(2 * k, h1), nn.ReLU(),
                                 nn.Linear(h1, h2), nn.ReLU(),
                                 nn.Linear(h2, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, h2), nn.ReLU(),
                                 nn.Linear(h2, h1), nn.ReLU(),
                                 nn.Linear(h1, k))

    def forward(self, x, m):
        return self.dec(self.enc(torch.cat([x * (1 - m), m], 1)))

    def embed(self, x):
        m = torch.zeros_like(x)
        return self.enc(torch.cat([x, m], 1))


def pretrain(X, seed, device='cpu', epochs=EPOCHS):
    """X: (n, k) already standardised. Returns the fitted encoder and its loss."""
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    n, k = X.shape
    cut = g.permutation(n)
    nval = max(64, int(n * VAL_FRAC))
    tr = torch.tensor(X[cut[nval:]], dtype=torch.float32, device=device)
    va = torch.tensor(X[cut[:nval]], dtype=torch.float32, device=device)
    # the validation mask is drawn once so the early-stopping signal is stable
    vm = (torch.rand(va.shape, generator=torch.Generator(device=device).manual_seed(seed),
                     device=device) < MASK_RATE).float()

    model = MaskedAutoencoder(k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), device=device)
        for i in range(0, len(tr), BATCH):
            xb = tr[perm[i:i + BATCH]]
            mb = (torch.rand_like(xb) < MASK_RATE).float()
            if mb.sum() == 0:
                continue
            loss = (((model(xb, mb) - xb) ** 2) * mb).sum() / mb.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float((((model(va, vm) - va) ** 2) * vm).sum() / vm.sum())
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best, ep + 1


def embed(model, X, device='cpu'):
    with torch.no_grad():
        return model.embed(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
