# projet_protannotml

Code of the ProtAnnotML capstone (course notebooks pf01-pf06). The `protannotml/` package is exported from the notebooks.

~~~
protannotml/   imgt.py (numbering, IMGT/AIRR readers) · synth.py (synthetic data) · splits.py · align.py (alignment baseline)
               features.py · decode.py (constrained Viterbi) · stats.py (evaluation) · labels.py · crf.py · model.py
               data.py · inference.py
scripts/       make_synthetic_data.py · smoke_test.py · dapt_mlm.py · train_multitask.py · annotate.py · serve.py
configs/       gpu6gb.yaml · v100.yaml · h100.yaml
slurm/         h100_train.slurm · v100_train.slurm · h100_dapt.slurm · chain.sh · prefetch_models.sh
~~~

Quick start (any machine with PyTorch):

~~~bash
pip install -r requirements.txt
python scripts/make_synthetic_data.py          # writes data_protannotml/ (fictional sequences)
python scripts/smoke_test.py                   # 1-2 min on CPU, must print SMOKE TEST PASSED
python scripts/train_multitask.py --config configs/gpu6gb.yaml --override epochs=1
~~~

Jean Zay: run `slurm/prefetch_models.sh` on a login node, then `bash slurm/chain.sh slurm/h100_train.slurm 3`.
Replace every `CHANGE_ME` (account, module version) first.
