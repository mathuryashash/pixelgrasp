"""Train the BC policy. Validation = held-out demo episodes (loss) + a closed-loop success check every N epochs.

Usage: python -m pixelgrasp.train --data data/demos --out runs/bc_k1 --chunk 1 --epochs 20
       python -m pixelgrasp.train --data data/demos --out runs/bc_k8 --chunk 8 --exec-steps 4 --epochs 20
"""
from __future__ import annotations
import argparse, glob, json, os, time
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from .policy import BCPolicy, BCPolicyRunner, loss_fn
from .eval import evaluate
from .env import DEV_SEED_BASE, DEMO_SEED_BASE


class DemoDataset(Dataset):
    def __init__(self, data_dir, chunk, episodes=None, augment=False, use_goal=False):
        shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
        parts = []
        for s in shards:                      # filter per shard BEFORE concatenating: peak RAM = final size, not 2x
            x = np.load(s); keep = np.isin(x["episode"], episodes) if episodes is not None else slice(None)
            if episodes is not None and not keep.any(): continue
            parts.append({k: x[k][keep] for k in x.files})
        D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        del parts
        order = np.lexsort((D["t"], D["episode"]))
        if not np.array_equal(order, np.arange(len(order))):
            for k in D: D[k] = D[k][order]
        if use_goal:
            assert "goal" in D, "dataset has no 'goal' key; re-run collect.py"
            D["proprio"] = np.concatenate([D["proprio"], D["goal"]], 1)
        self.img, self.proprio, self.action, self.ep, self.t = D["img"], D["proprio"], D["action"], D["episode"], D["t"]
        # index of next step within the same episode for chunk building
        n = len(self.t); self.chunk = chunk; self.augment = augment
        self.ep_end = np.zeros(n, np.int64)
        starts = np.flatnonzero(np.r_[True, self.ep[1:] != self.ep[:-1]]); ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends): self.ep_end[s:e] = e
        self.p_mean, self.p_std = self.proprio.mean(0), self.proprio.std(0) + 1e-6

    def __len__(self): return len(self.t)

    def __getitem__(self, i):
        K = self.chunk; e = self.ep_end[i]; idx = np.minimum(np.arange(i, i + K), e - 1)
        mask = (np.arange(i, i + K) < e).astype(np.float32)
        act = self.action[idx]
        img = self.img[i]
        if self.augment:   # random shift by up to 4 px (pad + crop), the standard BC image aug
            pad = 4; H = img.shape[0]
            p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            dx, dy = np.random.randint(0, 2 * pad + 1, 2); img = p[dy:dy + H, dx:dx + H]
        return torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(self.proprio[i]), torch.from_numpy(act), torch.from_numpy(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/demos"); ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1); ap.add_argument("--exec-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-eps", type=int, default=100); ap.add_argument("--max-train-eps", type=int, default=None)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--goal", action="store_true", help="give the policy the goal xyz (task command); cube still from pixels")
    ap.add_argument("--eval-every", type=int, default=5); ap.add_argument("--eval-seeds", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=0, help="also keep ep_NNN.pt every N epochs (for progress videos)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    assert meta["seed_base"] >= DEMO_SEED_BASE, f"demo seeds must start at >= {DEMO_SEED_BASE}, got {meta['seed_base']}"
    all_eps = np.arange(meta["seed_base"], meta["seed_base"] + meta["episodes_requested"])
    rng = np.random.default_rng(0); rng.shuffle(all_eps)
    val_eps, train_eps = all_eps[: a.val_eps], all_eps[a.val_eps:]
    if a.max_train_eps: train_eps = train_eps[: a.max_train_eps]
    tr = DemoDataset(a.data, a.chunk, train_eps, augment=a.augment, use_goal=a.goal); va = DemoDataset(a.data, a.chunk, val_eps, use_goal=a.goal)
    print(f"train {len(tr)} steps / {len(np.unique(tr.ep))} eps | val {len(va)} steps / {len(np.unique(va.ep))} eps | device {dev} | goal={a.goal}")
    model = BCPolicy(chunk=a.chunk, pretrained=True, use_goal=a.goal).to(dev)
    model.p_mean.copy_(torch.from_numpy(tr.p_mean)); model.p_std.copy_(torch.from_numpy(tr.p_std))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.bs + 1); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    dl = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=False, pin_memory=dev == "cuda")
    dlv = DataLoader(va, batch_size=512, shuffle=False, num_workers=0)
    log, best = [], (-1.0, 1e9); t0 = time.time()
    exec_steps = a.exec_steps or a.chunk
    for ep in range(1, a.epochs + 1):
        model.train(); tl = tp = tg = 0.0; n = 0
        for img, pro, act, mask in dl:
            img, pro, act, mask = img.to(dev, non_blocking=True), pro.to(dev), act.to(dev), mask.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                pos, g = model(img, pro)
            loss, lp, lg = loss_fn(pos.float(), g.float(), act, mask)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(img); tp += lp.item() * len(img); tg += lg.item() * len(img); n += len(img)
        model.eval(); vl = 0.0; vn = 0
        with torch.no_grad():
            for img, pro, act, mask in dlv:
                pos, g = model(img.to(dev), pro.to(dev)); loss, _, _ = loss_fn(pos, g, act.to(dev), mask.to(dev)); vl += loss.item() * len(img); vn += len(img)
        row = {"epoch": ep, "train_loss": tl / n, "train_pos": tp / n, "train_grip": tg / n, "val_loss": vl / vn, "lr": sched.get_last_lr()[0], "wall_s": round(time.time() - t0)}
        ck = {"state_dict": model.state_dict(), "chunk": a.chunk, "exec_steps": exec_steps, "use_goal": a.goal, "args": vars(a), "epoch": ep}
        torch.save(ck, os.path.join(a.out, "last.pt"))
        if a.save_every and ep % a.save_every == 0: torch.save(ck, os.path.join(a.out, f"ep_{ep:03d}.pt"))
        if ep % a.eval_every == 0 or ep == a.epochs:
            runner = BCPolicyRunner(os.path.join(a.out, "last.pt"), device=dev)
            s, _ = evaluate(runner, a.eval_seeds, seed_base=DEV_SEED_BASE)
            row["dev_success"] = s["success_rate"]; row["dev_causes"] = s["causes"]
            if (s["success_rate"], -row["val_loss"]) > (best[0], -best[1]):
                best = (s["success_rate"], row["val_loss"]); torch.save(ck, os.path.join(a.out, "best.pt"))
        log.append(row); print(json.dumps(row)); json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    print("best dev success", best[0])


if __name__ == "__main__":
    main()
