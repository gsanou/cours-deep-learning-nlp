"""Exported from notebook pf04_modele_plm.ipynb — edit the notebook source (src/pf04_modele_plm.md), not this file."""

"""Multitask fine-tuning of ProtAnnotML.

Single GPU : python scripts/train_multitask.py --config configs/gpu6gb.yaml
Multi-GPU  : torchrun --nproc_per_node=4 scripts/train_multitask.py --config configs/h100.yaml
Slurm      : srun python scripts/train_multitask.py --config configs/h100.yaml   (one task per GPU, see slurm/)
"""
import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protannotml.data import Collator, EncodedDataset          # noqa: E402
from protannotml.labels import LabelSpace, encode_example        # noqa: E402
from protannotml.model import ProtAnnotModel                     # noqa: E402
from protannotml.synth import load_jsonl                         # noqa: E402

STOP = {"flag": False}


def on_signal(signum, frame):                                    # Slurm sends SIGUSR1 before the time limit (--signal)
    STOP["flag"] = True


def setup_distributed():
    env = os.environ
    if "WORLD_SIZE" not in env and int(env.get("SLURM_NTASKS", "1")) > 1:   # launched by srun, one task per GPU
        env["WORLD_SIZE"], env["RANK"], env["LOCAL_RANK"] = env["SLURM_NTASKS"], env["SLURM_PROCID"], env["SLURM_LOCALID"]
    if "WORLD_SIZE" in env and int(env["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return rank, dist.get_world_size(), torch.device("cuda", local)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def apply_lora(model, cfg):
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg.get("lora_dropout", 0.05),
                      target_modules=["query", "key", "value", "dense"], bias="none")
    model.backbone = get_peft_model(model.backbone, lcfg)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="key=value pairs, e.g. lr_backbone=1e-5")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for kv in args.override:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)

    rank, world, device = setup_distributed()
    set_seed(cfg.get("seed", 0) + rank)
    signal.signal(signal.SIGUSR1, on_signal)
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)

    examples = {e["id"]: e for e in load_jsonl(cfg["examples"])}
    split = json.load(open(cfg["splits"]))[cfg["regime"]]
    train_ex, val_ex = [examples[i] for i in split["train"]], [examples[i] for i in split["val"]]
    space = LabelSpace.from_examples(train_ex)
    if rank == 0:
        space.save(out / "label_space.json")
        yaml.safe_dump(cfg, open(out / "config.yaml", "w"))
    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    train_ds = EncodedDataset([encode_example(e, space, cfg["max_residues"]) for e in train_ex])
    val_ds = EncodedDataset([encode_example(e, space, cfg["max_residues"]) for e in val_ex])
    sampler = DistributedSampler(train_ds, shuffle=True) if world > 1 else None
    collate = Collator(tok)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, shuffle=sampler is None,
                          collate_fn=collate, num_workers=cfg.get("num_workers", 4), pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], collate_fn=collate, num_workers=2)

    model = ProtAnnotModel(space.sizes(), backbone=cfg["backbone"], gradient_checkpointing=cfg.get("gradient_checkpointing", False))
    if cfg.get("lora", False):
        model = apply_lora(model, cfg)
    model.to(device)
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": cfg["lr_backbone"]},
                             {"params": head_params, "lr": cfg["lr_heads"]}], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_dl) / cfg["grad_accum"])
    total_steps, warmup = steps_per_epoch * cfg["epochs"], int(0.06 * steps_per_epoch * cfg["epochs"])
    def lr_lambda(s):                                                       # linear warmup, then linear decay
        if s < warmup:
            return (s + 1) / max(1, warmup)
        return max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    use_cuda = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and amp_dtype == torch.float16)
    ddp_model = DDP(model, device_ids=[device.index]) if world > 1 else model

    start_epoch, step = 0, 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():                                                   # resume
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start_epoch, step = ck["epoch"], ck["step"]
        if rank == 0:
            print(f"resumed from epoch {start_epoch}, step {step}", flush=True)

    def save(epoch):
        if rank == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), epoch=epoch, step=step), out / "last.tmp")
            os.replace(out / "last.tmp", ckpt_path)                         # atomic: never a half-written checkpoint

    def should_stop():
        """Collective decision: every rank must leave the loop at the same step, otherwise DDP hangs."""
        flag = torch.tensor([1.0 if STOP["flag"] else 0.0], device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def to_device(batch):
        return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    log = open(out / "log.jsonl", "a") if rank == 0 else None
    for epoch in range(start_epoch, cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ddp_model.train()
        t0 = time.time()
        for it, batch in enumerate(train_dl):
            batch = to_device(batch)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                res = ddp_model(**batch)
                loss = res["loss"] / cfg["grad_accum"]
            scaler.scale(loss).backward()
            if (it + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if rank == 0 and step % cfg.get("log_every", 50) == 0:
                    log.write(json.dumps(dict(epoch=epoch, step=step, loss=float(res["loss"]), **res["losses"],
                                              weights=torch.exp(-model.log_vars.detach()).tolist())) + "\n"); log.flush()
                if should_stop():
                    STOP["flag"] = True
                    break
        save(epoch + (0 if STOP["flag"] else 1))
        if world > 1:
            dist.barrier()                                                   # nobody continues before the checkpoint exists
        if STOP["flag"]:
            if rank == 0:
                print("stopping on signal; checkpoint saved, resubmit the job to resume", flush=True)
            break
        # quick validation: token-level accuracies (full decoding: protannotml.inference.annotate, scripts/annotate.py)
        model.eval(); correct = total = bio_ok = bio_tot = 0
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
            for batch in val_dl:
                batch = to_device(batch)
                res = model(batch["input_ids"], batch["attention_mask"], batch["residue_mask"])
                m = batch["pos"] != -100
                correct += (res["pos_logits"].argmax(-1)[m] == batch["pos"][m]).sum().item(); total += m.sum().item()
                rm = batch["residue_mask"]
                bio_ok += (res["bio_logits"].argmax(-1)[rm] == batch["bio"][rm]).sum().item(); bio_tot += rm.sum().item()
        if rank == 0:
            log.write(json.dumps(dict(epoch=epoch, val_pos_acc=correct / max(1, total), val_bio_acc=bio_ok / max(1, bio_tot),
                                      epoch_minutes=(time.time() - t0) / 60)) + "\n"); log.flush()
            print(f"epoch {epoch}: val position acc {correct / max(1, total):.4f}, bio acc {bio_ok / max(1, bio_tot):.4f}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
