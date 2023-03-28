# Results

| 例子 | 优化目标 | 采样个数 M | 网络层数 | 神经元个数 | 学习率 |
| FuSinCos | $\gamma$-BML | 4096 | 2 | 8 | 0.001 |
| | $\lambda$-BML | 4096 | 2 | 8 | 0.001 |
| | $\delta$-BML | 4096 | 2 | 8 | 0.001 |
| LongSin | $\gamma$-BML | 1024 | 3 | 32 | 0.001 |
| | $\lambda$-BML | 1024 | 3 | 32 | 0.001 |
| | $\delta$-BML | 1024 | 3 | 32 | 0.001 |
| JiLQ EXP1 | $\gamma$-BML | 64 | 2 | 16 | 0.001 |
| | $\lambda$-BML | 64 | 2 | 16 | 0.001 |
| | $\delta$-BML | 64 | 2 | 16 | 0.001 |
| JiLQ EXP2 | $\gamma$-BML | 64 | 2 | 16 | 0.002 |
| | $\lambda$-BML | 64 | 2 | 16 | 0.002 |
| | $\delta$-BML | 64 | 2 | 16 | 0.0005 |

## FuSinCos EXP1

- `dirac=False`
  - log_dir: outputs/230328-0056
  - commit ID: 0c05259
- `dirac=True`
  - commit ID: 7699a80
- `dirac=0.05`
  - log_dir: outputs/230328-0132
  - commit ID: 6622ac1

## LongSinCos EXP1

- `dirac=False`
  - log_dir: outputs/230329-0003
  - commit ID: a21e2e9
- `dirac=True` (STRANGE RESULT!)
  - log_dir: outputs/230329-0017
  - commit ID: d94da2c
- `dirac=0.05`
  - log_dir: outputs/230329-0011
  - commit ID: 752f8be

## JiLQ EXP1

- `dirac=False`
  - log_dir: outputs/230329-0046
  - commit ID: e4625c8
- `dirac=True`
  - log_dir: outputs/230329-0049
  - commit ID: 0faf7ae
- `dirac=0.05`
  - log_dir: outputs/230328-2036
  - commit ID: f8606f6

## JiLQ EXP2

- `dirac=False`
  - log_dir: outputs/230328-2051
  - commit ID: 32c0e77
- `dirac=True`
  - log_dir: outputs/230328-2059
  - commit ID: 1728a62
- `dirac=0.05`
  - log_dir: outputs/230329-0114
  - commit ID: 41771e9
