#!/bin/bash
# Run on a Jean Zay login node (internet access), NOT in a job: fills the local Hugging Face cache.
set -euo pipefail
export HF_HOME=$WORK/hf_cache
for m in facebook/esm2_t12_35M_UR50D facebook/esm2_t30_150M_UR50D facebook/esm2_t33_650M_UR50D; do
  python -c "from transformers import AutoTokenizer, EsmForMaskedLM; AutoTokenizer.from_pretrained('$m'); EsmForMaskedLM.from_pretrained('$m')"
done
du -sh $HF_HOME
