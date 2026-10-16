# FS12 联邦学习实验 — RunPod 交接手册 (AI-readable handoff)

> 目的:把这份文件 + `fl_bundle_full.tar.gz` 交给任何 AI 或新会话,即可在一个全新的
> RunPod GPU pod 上从零把实验跑起来、监控、并把结果取回本地。所有命令可直接复制执行。

---

## 0. 一句话总览

在 RunPod 的 CUDA GPU pod 上,对 **13 个 QLD 小时级空气质量监测站** 做联邦学习,预测 PM2.5。
跑 `{Transformer, Mamba} × {FedAvg, FedProx}` 共 4 个实验,每个实验 30 轮通信 (rounds),
记录**每轮平均 test MAE、收敛轮数、每轮耗时、单实验总时长**。

- 每个"客户端" = 一个监测站(数据不出本地,只交换模型权重)。
- 用近 5 年数据 (2021–2025),168 小时回看窗口,预测未来 24 小时。
- 本地代码在 `/Users/wangshihan/Air Quality Research/FreEformer/`;上传包见第 2 节。

---

## 1. 需要什么 (Requirements)

| 项目 | 值 |
|---|---|
| Pod 类型 | 任意 CUDA GPU;已验证 **RTX 4090** ($0.75/hr) 与 RTX PRO 4000 Blackwell 可跑 |
| 显存 | ≥16 GB 足够(用了 on-demand GPU:同一时刻只有 1 个模型在显存里) |
| 磁盘 | ≥20 GB workspace |
| torch | **必须 CUDA 版**;Blackwell (sm_120) 需 `cu128`(见第 4 节) |
| Python | pod 自带 3.10–3.12 均可 |
| Mamba 实验额外需 | `pip install mamba-ssm causal-conv1d`(仅 CUDA;Transformer 实验不需要) |

Transformer 的 4 个 pip 依赖:`timm reformer_pytorch seaborn` + torch/torchvision/numpy/pandas/scikit-learn(见 `requirements.txt`,bundle 里有)。

---

## 2. 上传包 (Bundle)

本地路径:`/Users/wangshihan/Air Quality Research/fl_bundle_full.tar.gz`(约 9.9 MB)

解包后得到一个完整的 `FreEformer/` 目录,关键内容:
```
FreEformer/
├─ run.py                         # (原始入口,FL 不直接用)
├─ requirements.txt
├─ model/                         # FrePatchTST3_fre_patch / _temporal / _mamba / _fusion
├─ layers/  utils/  experiments/
├─ data_provider/                 # data_factory.py + data_loader.py(每站一个 Dataset)
└─ federated/                     # ★ FL 全部代码
   ├─ run_federated.py            #   主编排器(下面所有命令都调它)
   ├─ client.py                   #   一个站=一个 Client;on-demand GPU 搬运
   ├─ aggregation.py              #   fedavg / fedavg_personalized / freqband
   ├─ fed_config.py               #   STATIONS_HOURLY(13站) + HOURLY_OVERRIDES(enc_in=13)
   ├─ run_fs12_experiments.sh     #   一键跑 4 个实验的脚本
   └─ HANDOFF.md                  #   本文件
Finalized_Dataset_hourly_common/  # ★ 13 个站的 csv(与 FreEformer/ 同级)
   ├─ qld_cannon_hill.csv ... qld_woolloongabba.csv   (共 13 个)
```

**数据列**(每个 csv):
`date, dbt, pm10, swd, sws, sigma60, relative_humidity, barometric_pressure_hpa, wind_speed_std_dev, no2, no, nox, rain, pm25`
→ 12 个协变量 + 目标 `pm25`(最后一列),MS 模式,`enc_in=13`。负值已 clip 到 0。

> ⚠️ 目录结构关键点:`Finalized_Dataset_hourly_common/` 必须和 `FreEformer/` **同级**
> (代码里用相对路径 `../Finalized_Dataset_hourly_common/`)。bundle 已按此结构打好。

---

## 3. 在 Pod 上部署(解包)

在 Jupyter Terminal 里(假设 bundle 已上传到 `/workspace/`):
```bash
cd /workspace
tar --no-same-owner -xzf fl_bundle_full.tar.gz    # 解出 FreEformer/ 和 Finalized_Dataset_hourly_common/
ls FreEformer/federated/run_federated.py          # 确认存在
ls Finalized_Dataset_hourly_common/*.csv | wc -l  # 应输出 13
```

---

## 4. 装依赖

```bash
cd /workspace/FreEformer
pip install -r requirements.txt
pip install timm reformer_pytorch seaborn        # 若 requirements 里没带全

# —— 仅当 torch 不是 CUDA 版 / 是 Blackwell(sm_120) 卡时才需要重装 torch ——
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# 若 cuda 为 False 或 capability=(12,0) 但报 sm_120 unsupported:
pip install --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
验证 GPU 可用:
```bash
python -c "import torch;print('GPU OK' if torch.cuda.is_available() else 'NO GPU', torch.__version__)"
```

---

## 5. 启动实验 (核心命令)

**从 `/workspace/FreEformer` 目录运行。** 一条命令跑 Transformer 的 fedavg + fedprox:

```bash
cd /workspace/FreEformer
nohup bash -c 'for m in fedavg fedprox; do
  echo "=== transformer/$m ===";
  python federated/run_federated.py \
    --mode $m \
    --backbone FrePatchTST3_fre_patch \
    --granularity hourly \
    --seq_len 168 --pred_len 24 \
    --full_size \
    --batch_size 128 --num_workers 8 --window_stride 12 \
    --rounds 30 --local_epochs 1 --seed 2023 \
    --eval_every 5 --skip_existing;
done' > tf.log 2>&1 &
echo "PID=$!  ->  tail -f /workspace/FreEformer/tf.log"
```

Mamba 的两个实验(需先 `pip install mamba-ssm causal-conv1d`),把 `--backbone` 换成 `FrePatchTST3_mamba`,输出改到 `mamba.log`。或者直接用一键脚本跑全部 4 个:
```bash
bash federated/run_fs12_experiments.sh          # transformer+mamba × fedavg+fedprox
```

### 参数含义(改动前务必理解)
| 参数 | 值 | 说明 / 能不能改 |
|---|---|---|
| `--mode` | `fedavg` / `fedprox` | 聚合方式。fedprox 额外用 `--mu 0.01` 近端项 |
| `--backbone` | `FrePatchTST3_fre_patch`(=Transformer) / `FrePatchTST3_mamba` | 频域+注意力 / 频域+Mamba |
| `--granularity hourly` | 固定 | 切到 13 站小时数据 + enc_in=13(HOURLY_OVERRIDES) |
| `--seq_len 168 --pred_len 24` | 固定 | 回看 7 天,预测 1 天 |
| `--full_size` | 开 | 论文规模 backbone(d_model=512,e_layers=2) |
| `--batch_size 128` | 可调 | **数据不多,别调太大**,会影响预测质量 |
| `--window_stride 12` | 可调 | 每 12 个重叠窗口取 1 个,砍冗余计算。**不影响精度逻辑**,只影响训练窗口数 |
| `--rounds 30` | 可调 | 通信轮数。减少会**略微**降精度 |
| `--eval_every 5` | 可调 | 每 5 轮评估一次记收敛曲线。**对精度零影响**,只影响日志频率。想每轮都看设 1 |
| `--seed 2023` | 记录用 | 换 seed 结果进不同子目录,不会覆盖 |
| `--skip_existing` | 开 | 断点续跑:该配置 13 站都跑完就跳过 |

> 打印规律:开了 `--eval_every 5` 后,**只有第 1、5、10…30 轮打印**,中间轮静默。
> 第 1 轮最慢(含一次性加载 13 站数据,约 176s on 4090),之后每轮更快。

---

## 6. 监控(实验跑起来后)

看跑到第几轮 + 日志是否在更新(判断是否卡住):
```bash
grep -oE 'round [0-9]+/30' /workspace/FreEformer/tf.log | tail -1
echo "日志 $(($(date +%s)-$(stat -c %Y /workspace/FreEformer/tf.log)))秒前更新"
tail -15 /workspace/FreEformer/tf.log
```
判断:
- **轮次在涨 / 日志几十秒内更新过** → 正常,继续等。
- **GPU 偶尔掉到 0%** → 正常!那是轮次之间的 CPU 环节(聚合权重 + 模型 CPU↔GPU 搬运),几秒后回 100%。
- **只有** 日志几百秒不动 **且** CPU 长时间为 0 → 才是真卡住(见第 8 节)。

GPU 占用(可选):`nvidia-smi` 或 `watch -n2 nvidia-smi`。

---

## 7. 取结果回本地

结果 JSON 路径(pod 上):
```
FreEformer/federated/results/hourly/sl168_pl24/seeds/s2023/
  ├─ Fed-fedavg-fre_<station>_metrics.json      # 每站最终 MAE/RMSE/R²
  ├─ Fed-fedavg-fre_convergence.json            # ★ 收敛曲线:每轮MAE、converged_round、avg_round_sec、total_min
  ├─ Fed-fedprox-fre_*_metrics.json
  └─ Fed-fedprox-fre_convergence.json
```
`*_convergence.json` 就是要交论文的关键(几轮收敛 + 每轮/总耗时)。

打包下载:
```bash
cd /workspace/FreEformer/federated
tar -czf /workspace/fs12_results.tar.gz results/hourly/
# 然后在 Jupyter 文件树里右键 fs12_results.tar.gz -> Download,或用 runpodctl send
```

---

## 8. 故障排查 (Troubleshooting)

| 症状 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | 13 个模型都进显存了 | 确认用的是**含 on-demand GPU 修复的 bundle**(`client.py` 里有 3 处 `_free_gpu`);再不行 `--batch_size 64` |
| `sm_120 not supported` / `torchvision::nms does not exist` | torch 非 cu128 | 见第 4 节 force-reinstall cu128 |
| NVML `INTERNAL ASSERT ... CUDACachingAllocator` | MIG 切片卡的 bug | 换非-MIG 整卡(4090 正常);别用 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`(会 OOM/变慢) |
| `ModuleNotFoundError: seaborn/timm/reformer_pytorch` | 依赖没装 | `pip install seaborn timm reformer_pytorch` |
| `mamba_ssm` 报错 | Mamba 依赖缺失 | `pip install mamba-ssm causal-conv1d`(仅 CUDA) |
| 日志几百秒不动 + CPU=0(真卡住) | 可能 DataLoader worker 卡死 | kill 进程,重启时把 `--num_workers 8` 改成 `--num_workers 0` |
| `tar: unexpected end of file` | bundle 上传不完整 | 重新上传;解包加 `--no-same-owner` |
| 想同时跑多个实验 | 单卡显存有限 | 不建议同卡并行(抢显存反而更慢);要并行就开第二个 pod 各跑一半 |

验证 bundle 完整性(有没有 on-demand GPU 修复):
```bash
grep -c "_free_gpu" /workspace/FreEformer/federated/client.py   # 应为 3
ls /workspace/FreEformer/data_provider/data_factory.py          # 应存在
```

---

## 9. 现在这个实验跑完后要做什么

1. 确认 4 个 `*_convergence.json` 都生成了。
2. `tar` 打包 `results/hourly/` 下载到本地 `FreEformer/federated/results/hourly/` 同名路径。
3. 汇总成论文表:每个 (backbone×mode) 的 **converged_round / avg_round_sec / total_min / best_avg_mae**,
   以及 13 站平均的最终 MAE/RMSE/R²。
4. 关掉 pod(省钱)。

---

## 10. 关键设计备忘(给接手的 AI)

- **on-demand GPU**:`client.py` 里模型默认在 CPU,`local_train`/`evaluate` 开头 `.to(device)`,
  结束 `_free_gpu()` 搬回 CPU + `empty_cache()`。这样峰值显存只有 1 个模型,是解决 13 站 OOM 的根本修复。**不要改回一次性全放 GPU。**
- **评估指标**:`test_r2` 是标准全局 R²(sklearn),`test_r2_perwindow` 是原 FreEformer 的逐窗口 R²(对平滑信号过于苛刻,仅参考)。报论文用 `test_r2`。
- **频域 vs 时域故事**:小时数据有强 24h 日周期(lag24≈0.4,FFT 在 24h 有峰),支撑"频域模型更鲁棒"的卖点。
- 更多背景见本地记忆:`.claude/.../memory/fl-air-quality-research.md` 与 `fl-harness-and-conventions.md`。
