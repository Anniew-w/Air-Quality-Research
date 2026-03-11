# RunPod 跑模型说明

这份仓库里的单站点训练脚本，当前最适合直接放到 RunPod 上跑的是下面几个：

- `Models/tcn_running.py`
- `Models/lstm_running.py`
- `Models/gnn_transformer_running.py`
- `Models/pgn_running.py`

`Models/modernTCN_running.py` 目前不建议直接上 RunPod，因为仓库里缺少 `Models/ModernTCN.py` 源文件，只剩下 `Models/__pycache__/ModernTCN.cpython-38.pyc`。换机器或换 Python 版本时很容易 import 失败。

## 1. RunPod 机器建议

如果只是跑你现在这些单站点脚本，优先选：

- `PyTorch` 模板
- 1 块 NVIDIA GPU
- 30GB 以上磁盘

这些脚本数据不算大，GPU 显存压力通常不会特别高，先从便宜卡开始就够了。

## 2. 上传项目

在 RunPod Terminal 里把仓库放进去，例如：

```bash
cd /workspace
git clone <你的仓库地址> air-quality-research
cd air-quality-research
```

如果你不是用 git，也可以直接把本地项目压缩后上传到 `/workspace` 再解压。

## 3. 安装依赖

```bash
cd /workspace/air-quality-research
python -m pip install -U pip
pip install -r requirements-runpod.txt
```

如果 RunPod 模板自带了 PyTorch，这一步会跳过已有包；如果没有，就会安装 CPU/GPU 版本里兼容的 `torch`。

## 4. 直接运行

现在这些脚本已经支持命令行传参，不再依赖你本机绝对路径。

例子：

```bash
cd /workspace/air-quality-research
python Models/tcn_running.py --data Finalized_Dataset/Melbourne_CBD_10239_merged.csv --epochs 50
python Models/lstm_running.py --data Finalized_Dataset/Box_Hill_10042_merged.csv --epochs 80
python Models/gnn_transformer_running.py --data Finalized_Dataset/Traralgon_10011_merged.csv --epochs 40
python Models/pgn_running.py --data Finalized_Dataset/Melton_10169_merged.csv --epochs 40
```

如果你想自定义输出名字：

```bash
python Models/tcn_running.py \
  --data Finalized_Dataset/Melbourne_CBD_10239_merged.csv \
  --epochs 50 \
  --name melbourne_tcn_runpod
```

## 5. 结果在哪

这些脚本会把图保存到运行目录下的 `results/`。

如果你在仓库根目录执行：

```bash
cd /workspace/air-quality-research
python Models/tcn_running.py --data Finalized_Dataset/Melbourne_CBD_10239_merged.csv
```

那么结果通常会在：

```bash
/workspace/air-quality-research/results
```

可以这样打包下载：

```bash
tar -czf results.tar.gz results
```

## 6. 后台跑

为了避免终端断开后训练中止，建议用：

```bash
nohup python Models/tcn_running.py --data Finalized_Dataset/Melbourne_CBD_10239_merged.csv --epochs 50 > tcn.log 2>&1 &
tail -f tcn.log
```

其它模型同理。

## 7. 你现在这份仓库的实际注意点

- `Models/modernTCN_running.py` 还缺 `ModernTCN.py` 源码，先不要作为 RunPod 主力入口。
- 这几份脚本是“单个 CSV -> 单站点训练”，不是统一调度器；如果你要批量跑多个站点，最好再加一个 bash/python 批处理脚本。
- `results/` 是相对路径；建议始终从仓库根目录执行命令，避免输出散落到别的目录。

## 8. 最省事的推荐顺序

先在 RunPod 上验证下面这个命令：

```bash
python Models/tcn_running.py --data Finalized_Dataset/Melbourne_CBD_10239_merged.csv --epochs 10
```

确认能出图、能识别 GPU 后，再切到：

- `lstm_running.py`
- `pgn_running.py`
- `gnn_transformer_running.py`

如果你后面要，我可以继续帮你补两样东西：

1. 一个批量跑多个站点的 `run_all.sh`
2. 一个专门给 RunPod 用的 `Dockerfile`
