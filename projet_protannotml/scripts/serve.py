"""Exported from notebook pf06_echelle_deploiement.ipynb — edit the notebook source (src/pf06_echelle_deploiement.md), not this file."""

"""Minimal HTTP service. Run: PROTANNOT_RUN=runs/h100 uvicorn serve:app --app-dir scripts --port 8000"""
import os
import sys
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate import load_run                                 # noqa: E402
from protannotml.inference import annotate                    # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, TOK, SPACE = load_run(os.environ["PROTANNOT_RUN"], DEVICE)
app = FastAPI(title="ProtAnnotML", version="0.1.0")


class AnnotateRequest(BaseModel):
    sequences: list[str] = Field(..., max_length=256, description="protein sequences (one-letter code)")
    gene_threshold: float = 0.5


@app.post("/annotate")
def annotate_endpoint(req: AnnotateRequest):
    seqs = [s.strip().upper() for s in req.sequences]
    return annotate(MODEL, TOK, seqs, SPACE, device=DEVICE, gene_threshold=req.gene_threshold)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "genes": len(SPACE.genes)}
