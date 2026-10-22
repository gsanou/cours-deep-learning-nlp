"""Exported from notebook pf04_modele_plm.ipynb — edit the notebook source (src/pf04_modele_plm.md), not this file."""

"""Domain-adaptive MLM pretraining of an ESM-2 checkpoint on IG/TR sequences (one sequence per line).

Usage: python scripts/dapt_mlm.py --model facebook/esm2_t12_35M_UR50D --train ig_train.txt --valid ig_valid.txt --out runs/dapt35M
"""
import argparse
import os

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling, EsmForMaskedLM, Trainer,
                          TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=256)              # V domains and Fv chains are short
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_steps", type=int, default=50_000)
    ap.add_argument("--mlm_probability", type=float, default=0.15)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = EsmForMaskedLM.from_pretrained(args.model)
    ds = load_dataset("text", data_files={"train": args.train, "validation": args.valid})
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=args.mlm_probability)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()       # H100 / A100: bf16 ; V100: fp16
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, learning_rate=args.lr, warmup_ratio=0.05, weight_decay=0.01,
        max_steps=args.max_steps, lr_scheduler_type="cosine", bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=True, eval_strategy="steps", eval_steps=2000, save_steps=2000, save_total_limit=3,
        logging_steps=100, dataloader_num_workers=min(8, os.cpu_count() or 1), group_by_length=True,
        report_to="none", seed=0,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"], eval_dataset=ds["validation"], data_collator=collator)
    last = get_last_checkpoint(args.out) if os.path.isdir(args.out) else None
    trainer.train(resume_from_checkpoint=last)                       # resumes automatically after a Slurm time limit
    trainer.save_model(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
