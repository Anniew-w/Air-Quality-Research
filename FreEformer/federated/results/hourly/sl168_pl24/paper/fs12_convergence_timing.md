# FS12 convergence & timing

| Experiment            |   Converged round |   Best avg MAE |   Avg sec/round |   Total (h) |   Rounds |
|:----------------------|------------------:|---------------:|----------------:|------------:|---------:|
| Transformer + FedAvg  |                15 |         1.9049 |           167.8 |        1.67 |       30 |
| Transformer + FedProx |                15 |         1.9039 |           178.3 |        1.76 |       30 |
| Mamba + FedAvg        |                10 |         1.8992 |            34.8 |        0.35 |       30 |
| Mamba + FedProx       |                15 |         1.9024 |            38.5 |        0.39 |       30 |
