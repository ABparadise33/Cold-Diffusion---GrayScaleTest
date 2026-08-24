# Experiment log

Record real experiments here. Do not treat smoke tests as scientific evidence.

| Date | Question | Code/config | Data | Command | Result | Decision |
|---|---|---|---|---|---|---|
| 2026-08-24 | Does the implementation satisfy endpoint, sampler, and resume invariants? | Initial 50k implementation | Synthetic tensors and 8 synthetic paired images | `pytest -q`; smoke train 2 steps; `--resume auto --max-steps 4`; `evaluate.py` | 4/4 tests passed; CLI resumed from step 2 and completed step 4; evaluation artifacts written | Implementation is ready for a 50-pair overfit test; these results are not scientific evidence. |
