# RunPod GPU 运行手册 — 小时级联邦学习

任务：**多变量（FS8）小时级 pm25 预测**。24 个 QLD 客户端，8 个协变量
（dbt, pm10, swd, sws, sigma60, relative_humidity, barometric_pressure_hpa,
wind_speed_std_dev）+ pm25 目标，enc_in=9，lookback=168h → horizon=24h，MS 模式。
每站约 27k 训练 / 6.8k 测试窗口。

我（Claude）无法替你部署/付费 pod。你按下面几步做，pod 上基本只需复制粘贴。

## 0. 预算与 GPU 选择
- 你截图余额 **$4.27** —— 跑完整矩阵不够，需要充值，或先只跑核心子集（见第 5 步 A）。
- **推荐 GPU：RTX 4090（24GB，$0.74/hr）** 或 RTX 5090（$0.99/hr）。本模型 full_size d_model=512、batch 64、单变量，显存需求小，24GB 绰绰有余；瓶颈是**时间**（43站×3.3万窗口×多配置），不是显存。
- **不建议** H100/H200/B200（$3.29–6.79/hr，显存过剩、贵），除非想用速度换时间。
- 选 **PyTorch 官方模板**（自带 CUDA + torch），省装环境时间。

## 1. 部署 pod
Console → Pods → Deploy → 选 RTX 4090 → PyTorch 模板 → 部署。记下 SSH 连接命令（pod 页面 Connect 里有）。

## 2. 上传代码+数据（本地终端执行）
上传包已备好：`fl_hourly_bundle.tar.gz`（6.2MB，含精简代码 + 39MB 单变量数据）。
```bash
# 把 <POD_SSH> 换成 pod 的 scp 目标，如 root@213.xxx.xxx.xxx -p 12345
scp -P <PORT> fl_hourly_bundle.tar.gz root@<POD_IP>:/workspace/
```
或用 RunPod 网页版的文件上传 / `runpodctl send`。

## 3. pod 上解压 + 装依赖
```bash
cd /workspace && tar xzf fl_hourly_bundle.tar.gz
cd FreEformer && pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # 必须 True
# 可选：Mamba backbone 需要（仅这个才装，装不上就跳过，不影响其余）
# pip install mamba-ssm causal-conv1d
```

## 4. 冒烟测试（1个配置、2轮，确认能跑，约几分钟）
```bash
cd /workspace/FreEformer
python federated/run_federated.py --mode fedavg --backbone FrePatchTST3_fusion \
  --granularity hourly --seq_len 168 --pred_len 24 --full_size \
  --batch_size 64 --num_workers 8 --rounds 2 --local_epochs 1 --seed 2023
# 看到 11+ 个 eval 行 + "done in X min" 即成功
```

## 5. 正式跑
### A. 预算紧张时的核心子集（推荐先跑这个，1个seed）
只跑最能支撑论文的对比（fusion骨干 × 4种聚合 + 上下界 + 频/时对照）：
```bash
SEEDS="2023" bash federated/run_hourly_gpu.sh
```
### B. 完整（有预算/时间时）
```bash
SEEDS="2023 2024 2025" bash federated/run_hourly_gpu.sh
```
- 脚本 **resumable**（`--skip_existing`）：断了重跑会跳过已完成的，不重复烧钱。
- 建议 `tmux`/`nohup` 后台跑，断连不中断：
```bash
tmux new -s sweep 'SEEDS="2023" bash federated/run_hourly_gpu.sh 2>&1 | tee sweep.log'
# 断开: Ctrl-b d ; 重连: tmux attach -t sweep
```

## 6. 出表 + 下载结果
```bash
python federated/aggregate_federated.py --seeds_dir results/hourly/sl168_pl24/seeds
# 结果表在 results/hourly/sl168_pl24/paper_tables/
```
把结果打包下载回本地（结果很小，几 MB）：
```bash
tar czf hourly_results.tar.gz results/hourly
# 本地: scp -P <PORT> root@<POD_IP>:/workspace/FreEformer/hourly_results.tar.gz .
```
下载后本地放到 `FreEformer/federated/results/hourly/`，我就能帮你解读、整理成论文表格。

## ⚠️ 省钱提醒
- 跑完/挂起前 **Stop 或 Terminate pod**，否则一直计费。
- 先跑第 4 步冒烟 + 第 5A 核心子集，拿到第一批结果确认方向，再决定要不要扩到全 seed / 全 backbone。
