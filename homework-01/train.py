import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from model import Llama, LlamaConfig

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ROOT = Path(__file__).parent


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


def load_text(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding="utf-8")


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    starts = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


def lr_at(step: int, cfg: dict) -> float:
    # linear warmup, then cosine decay to min_lr
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: Llama, cfg: dict) -> torch.optim.AdamW:
    # no weight decay on norms
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg["lr"], betas=tuple(cfg["betas"]))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(model: Llama, splits: dict, cfg: dict, device: torch.device) -> dict:
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [
            model(*get_batch(data, cfg["batch_size"], cfg["seq_len"], device))[1].item()
            for _ in range(cfg["eval_batches"])
        ]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def check_model(model_cfg: LlamaConfig, device: torch.device) -> None:
    # causality: changing the last token must not affect earlier logits
    model = Llama(model_cfg).to(device).eval()
    tokens = torch.randint(model_cfg.vocab_size, (2, 32), device=device)
    changed = tokens.clone()
    changed[:, -1] = (changed[:, -1] + 1) % model_cfg.vocab_size
    with torch.no_grad():
        a, _ = model(tokens)
        b, _ = model(changed)
    assert a.shape == (2, 32, model_cfg.vocab_size)
    torch.testing.assert_close(a[:, :-1], b[:, :-1], rtol=1e-4, atol=1e-4)
    print("model check passed: output shape ok, attention is causal")


def overfit_single_batch(
    model_cfg: LlamaConfig, cfg: dict, data: torch.Tensor, device: torch.device, writer: SummaryWriter
) -> list[float]:
    torch.manual_seed(cfg["seed"])
    model = Llama(model_cfg).to(device)
    opt = make_optimizer(model, {**cfg, "weight_decay": 0.0})
    x, y = get_batch(data, 8, cfg["seq_len"], device)
    losses = []
    for step in range(cfg["overfit_steps"]):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, {**cfg, "max_steps": cfg["overfit_steps"], "warmup_steps": 20})
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        losses.append(loss.item())
        writer.add_scalar("overfit/loss", loss.item(), step)
        if step % 50 == 0 or step == cfg["overfit_steps"] - 1:
            print(f"[overfit] step {step:4d} | loss {loss.item():.4f}")
    return losses


def train(
    model_cfg: LlamaConfig, cfg: dict, splits: dict, device: torch.device, writer: SummaryWriter
) -> tuple[Llama, dict]:
    torch.manual_seed(cfg["seed"])
    model = Llama(model_cfg).to(device)
    print(f"parameters: {model.num_params() / 1e6:.2f}M")
    opt = make_optimizer(model, cfg)

    history = {"step": [], "train_loss": [], "lr": [], "grad_norm": [], "eval_step": [], "eval_train": [], "eval_val": []}
    start = time.time()
    for step in range(cfg["max_steps"]):
        lr = lr_at(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr

        if step % cfg["eval_interval"] == 0 or step == cfg["max_steps"] - 1:
            losses = estimate_loss(model, splits, cfg, device)
            history["eval_step"].append(step)
            history["eval_train"].append(losses["train"])
            history["eval_val"].append(losses["val"])
            writer.add_scalars("loss/eval", losses, step)
            print(f"step {step:5d} | eval train {losses['train']:.4f} | val {losses['val']:.4f} | {time.time() - start:.0f}s")

        x, y = get_batch(splits["train"], cfg["batch_size"], cfg["seq_len"], device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

        if step % cfg["log_interval"] == 0:
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["lr"].append(lr)
            history["grad_norm"].append(grad_norm.item())
            writer.add_scalar("loss/train_batch", loss.item(), step)
            writer.add_scalar("lr", lr, step)
            writer.add_scalar("grad_norm", grad_norm.item(), step)
    return model, history


def plot(history: dict, overfit_losses: list[float], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(history["step"], history["train_loss"], alpha=0.4, label="train (batch)")
    ax.plot(history["eval_step"], history["eval_train"], marker="o", ms=3, label="train (eval)")
    ax.plot(history["eval_step"], history["eval_val"], marker="o", ms=3, label="val (eval)")
    ax.set(title="Tiny Shakespeare: cross-entropy", xlabel="step", ylabel="loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(overfit_losses)
    ax.set(title="Overfit a single batch", xlabel="step", ylabel="loss", yscale="log")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history["step"], history["lr"])
    ax.set(title="Learning rate", xlabel="step")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(history["step"], history["grad_norm"])
    ax.set(title="Grad norm (before clipping)", xlabel="step")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"saved plot to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    train_cfg = config["train"]
    device = pick_device()
    print(f"device: {device}")

    text = load_text(ROOT / "data" / "tinyshakespeare.txt")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    splits = {"train": data[:split], "val": data[split:]}
    print(f"dataset: {len(data):,} chars, vocab {tokenizer.vocab_size}")

    model_cfg = LlamaConfig(vocab_size=tokenizer.vocab_size, **config["model"])
    check_model(model_cfg, device)

    args.out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.out / "tensorboard")
    writer.add_text("config", f"```json\n{json.dumps(config, indent=2)}\n```")

    overfit_losses = overfit_single_batch(model_cfg, train_cfg, splits["train"], device, writer)
    model, history = train(model_cfg, train_cfg, splits, device, writer)

    (args.out / "history.json").write_text(json.dumps({**history, "overfit_loss": overfit_losses}))
    torch.save({"model": model.state_dict(), "config": config}, args.out / "model.pt")
    plot(history, overfit_losses, args.out / "training.png")

    prompt = torch.tensor([tokenizer.encode("ROMEO:\n")], device=device)
    sample = model.eval().generate(prompt, max_new_tokens=400, temperature=0.8, top_k=20)
    sample_text = tokenizer.decode(sample[0].tolist())
    (args.out / "sample.txt").write_text(sample_text)
    writer.add_text("sample", f"```\n{sample_text}\n```")
    writer.close()
    print("\n--- sample ---\n" + sample_text)


if __name__ == "__main__":
    main()
