import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "llama.zip"))  # homework-01 code, unchanged

from model import Llama, LlamaConfig  # noqa: E402
from train import CharTokenizer, estimate_loss, get_batch, load_text, lr_at, make_optimizer  # noqa: E402

CHECK_STEPS = 5


def setup() -> tuple[torch.device, DeviceMesh]:
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device("cpu")  # MPS has no collective backend
        dist.init_process_group("gloo")
    # explicit mesh: the default one picks MPS on a Mac
    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    return device, mesh


def build_sharded(model_cfg: LlamaConfig, seed: int, device: torch.device, mesh: DeviceMesh) -> Llama:
    # same seed on every rank, so all shards come from one initialization
    torch.manual_seed(seed)
    model = Llama(model_cfg).to(device)
    # one FSDP unit per block (norm + attn + norm + mlp)
    for block in model.layers:
        fully_shard(block, mesh=mesh)
    # root unit: embeddings, final norm, output head
    fully_shard(model, mesh=mesh)
    return model


def mean_over_ranks(value: float, device: torch.device) -> float:
    t = torch.tensor(value, device=device)
    dist.all_reduce(t)
    return t.item() / dist.get_world_size()


def describe_sharding(model: Llama) -> str:
    units: dict[str, list[int]] = {}
    for name, param in model.named_parameters():
        unit = ".".join(name.split(".")[:2]) if name.startswith("layers.") else "root"
        global_numel, local_numel = units.setdefault(unit, [0, 0])
        units[unit] = [global_numel + param.numel(), local_numel + param.to_local().numel()]
    lines = [f"{'unit':10s} {'global':>10s} {'local':>10s}"]
    lines += [f"{unit:10s} {g:10,d} {l:10,d}" for unit, (g, l) in units.items()]
    total, local = (sum(v[i] for v in units.values()) for i in (0, 1))
    lines.append(f"{'total':10s} {total:10,d} {local:10,d}")
    return "\n".join(lines)


def check_against_single_process(
    model_cfg: LlamaConfig, cfg: dict, data: torch.Tensor, device: torch.device, mesh: DeviceMesh
) -> None:
    # FSDP over micro-batches must reproduce one unsharded model on the whole batch
    rank, world = dist.get_rank(), dist.get_world_size()
    micro = cfg["batch_size"] // world
    torch.manual_seed(cfg["seed"])
    reference = Llama(model_cfg).to(device)
    sharded = build_sharded(model_cfg, cfg["seed"], device, mesh)
    ref_opt, opt = make_optimizer(reference, cfg), make_optimizer(sharded, cfg)

    for step in range(CHECK_STEPS):
        torch.manual_seed(cfg["seed"] + step)  # same global batch on every rank
        x, y = get_batch(data, cfg["batch_size"], cfg["seq_len"], device)
        part = slice(rank * micro, (rank + 1) * micro)

        _, ref_loss = reference(x, y)
        ref_loss.backward()
        ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), cfg["grad_clip"])
        ref_opt.step()
        ref_opt.zero_grad(set_to_none=True)

        _, loss = sharded(x[part], y[part])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(sharded.parameters(), cfg["grad_clip"]).full_tensor()
        opt.step()
        opt.zero_grad(set_to_none=True)

        # grad norm catches sum-vs-mean reduction, which Adam alone would hide
        torch.testing.assert_close(mean_over_ranks(loss.item(), device), ref_loss.item(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(norm, ref_norm, rtol=1e-4, atol=1e-5)

    for (name, param), ref_param in zip(sharded.named_parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(param.full_tensor(), ref_param, rtol=1e-4, atol=1e-5, msg=name)
    if rank == 0:
        print(f"fsdp check passed: {CHECK_STEPS} steps match a single unsharded model on the full batch")


def train(
    model_cfg: LlamaConfig, cfg: dict, splits: dict, device: torch.device, mesh: DeviceMesh, writer: SummaryWriter | None
) -> tuple[Llama, dict]:
    rank, world = dist.get_rank(), dist.get_world_size()
    micro = cfg["batch_size"] // world
    model = build_sharded(model_cfg, cfg["seed"], device, mesh)
    opt = make_optimizer(model, cfg)
    if rank == 0:
        print(f"parameters: {model.num_params() / 1e6:.2f}M, world size {world}, micro-batch {micro}")
        print(describe_sharding(model))
    torch.manual_seed(cfg["seed"] + rank)  # different data on every rank

    history = {"step": [], "train_loss": [], "lr": [], "grad_norm": [], "eval_step": [], "eval_train": [], "eval_val": []}
    start = time.time()
    for step in range(cfg["max_steps"]):
        lr = lr_at(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr

        if step % cfg["eval_interval"] == 0 or step == cfg["max_steps"] - 1:
            local = estimate_loss(model, splits, {**cfg, "batch_size": micro}, device)
            losses = {name: mean_over_ranks(value, device) for name, value in local.items()}
            history["eval_step"].append(step)
            history["eval_train"].append(losses["train"])
            history["eval_val"].append(losses["val"])
            if rank == 0:
                writer.add_scalars("loss/eval", losses, step)
                print(f"step {step:5d} | eval train {losses['train']:.4f} | val {losses['val']:.4f} | {time.time() - start:.0f}s")

        x, y = get_batch(splits["train"], micro, cfg["seq_len"], device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).full_tensor()
        opt.step()

        if step % cfg["log_interval"] == 0:
            train_loss = mean_over_ranks(loss.item(), device)
            history["step"].append(step)
            history["train_loss"].append(train_loss)
            history["lr"].append(lr)
            history["grad_norm"].append(grad_norm.item())
            if rank == 0:
                writer.add_scalar("loss/train_batch", train_loss, step)
                writer.add_scalar("lr", lr, step)
                writer.add_scalar("grad_norm", grad_norm.item(), step)

    history["seconds"] = time.time() - start
    if device.type == "cuda":
        history["max_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return model, history


def plot(history: dict, baseline: dict | None, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(history["step"], history["train_loss"], alpha=0.4, label="FSDP train (batch)")
    ax.plot(history["eval_step"], history["eval_train"], marker="o", ms=3, label="FSDP train (eval)")
    ax.plot(history["eval_step"], history["eval_val"], marker="o", ms=3, label="FSDP val (eval)")
    if baseline:
        ax.plot(baseline["eval_step"], baseline["eval_val"], ls="--", color="gray", label="HW1 val, 1 process")
    ax.set(title="Tiny Shakespeare: cross-entropy", xlabel="step", ylabel="loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["step"], history["lr"])
    ax.set(title="Learning rate", xlabel="step")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(history["step"], history["grad_norm"], label="FSDP")
    if baseline:
        ax.plot(baseline["step"], baseline["grad_norm"], ls="--", color="gray", alpha=0.7, label="HW1, 1 process")
    ax.set(title="Grad norm (before clipping)", xlabel="step")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"saved plot to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--baseline", type=Path, default=ROOT.parent / "homework-01" / "outputs" / "history.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    train_cfg = config["train"]
    device, mesh = setup()
    rank, world = dist.get_rank(), dist.get_world_size()
    if train_cfg["batch_size"] % world != 0:
        raise ValueError(f"batch_size {train_cfg['batch_size']} is not divisible by world size {world}")

    text = load_text(ROOT / "data" / "tinyshakespeare.txt") if rank == 0 else None
    dist.barrier()  # rank 0 downloads, the rest wait
    text = text or (ROOT / "data" / "tinyshakespeare.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    splits = {"train": data[:split], "val": data[split:]}
    model_cfg = LlamaConfig(vocab_size=tokenizer.vocab_size, **config["model"])
    if rank == 0:
        print(f"device: {device}, backend: {dist.get_backend()}, world size: {world}")
        print(f"dataset: {len(data):,} chars, vocab {tokenizer.vocab_size}")

    check_against_single_process(model_cfg, train_cfg, splits["train"], device, mesh)

    writer = None
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.out / "tensorboard")
        writer.add_text("config", f"```json\n{json.dumps({**config, 'world_size': world}, indent=2)}\n```")
    model, history = train(model_cfg, train_cfg, splits, device, mesh, writer)

    # collective: every rank sends its shards, rank 0 gets the full tensors
    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        print(f"trained in {history['seconds']:.0f}s")
        (args.out / "history.json").write_text(json.dumps({**history, "world_size": world}))
        torch.save({"model": state, "config": config}, args.out / "model.pt")
        baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
        plot(history, baseline, args.out / "training.png")

        # the gathered checkpoint loads into a plain, unsharded Llama
        plain = Llama(model_cfg)
        plain.load_state_dict(state)
        prompt = torch.tensor([tokenizer.encode("ROMEO:\n")])
        sample = plain.eval().generate(prompt, max_new_tokens=400, temperature=0.8, top_k=20)
        sample_text = tokenizer.decode(sample[0].tolist())
        (args.out / "sample.txt").write_text(sample_text)
        writer.add_text("sample", f"```\n{sample_text}\n```")
        writer.close()
        print("\n--- sample ---\n" + sample_text)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
