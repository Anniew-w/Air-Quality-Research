# Air Quality Research 完整文件备份与恢复

备份日期：2026-10-16（Australia/Melbourne）。

GitHub：https://github.com/Anniew-w/Air-Quality-Research

## 换电脑或格式化后恢复

先安装 Git 和 Python 3，然后在新电脑终端执行：

```sh
git clone https://github.com/Anniew-w/Air-Quality-Research.git
cd Air-Quality-Research
python3 _backup_2026-10-16/verify_backup.py --restore-empty-dirs
```

Windows 可将 `python3` 换成 `python`。校验成功会显示 `PASS`，并逐一确认全部清单文件的 SHA-256。此备份直接保存文件本体，不需要 Git LFS，也不依赖原电脑或临时文件夹。

## 包含的内容

- 所有现存代码、Notebook、原始 Excel、EPA/BOM 数据与清洗中间结果。
- 全部 `Finalized_Dataset*`、`New_Dataset`、`outputs` 数据快照及其元数据。
- 根目录及子项目的模型权重、实验结果、图表、训练日志与源码快照。
- 已被子目录 `.gitignore` 忽略的现存研究文件。
- 原来只记录为 Git 外部引用的 `EPA_Backup/FelixOpolka_STGCN/STGCN-PyTorch` 目录内的实际文件。
- 现存 Python 字节码，包括少数已经没有对应源码的旧模型文件。
- `Reports` 报告、现有依赖说明、部署与实验交接文档。
- 原 Git 提交历史保留；备份前已从本机删除的已提交文件仍可从历史中取回。

`MODIFIED_FILE.json` 记录研究文件相对路径、字节数和 SHA-256；`directories` 记录空目录。只排除 Git 内部管理目录和本备份说明目录，避免清单自我引用。备份说明目录本身随 Git 提交保存。备份不会自动同步未来修改，后续研究修改仍需另行提交和推送。

## 恢复文件后运行实验

先按所需实验安装依赖，不要把所有独立项目混装到同一个环境。

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-runpod.txt
python Models/tcn_running.py --data Finalized_Dataset/Melbourne_CBD_10239_merged.csv --epochs 1
```

上面是恢复后的可选训练示例，不代表本次备份已训练或验证模型。FreEformer 联邦实验从 `FreEformer` 目录运行，以 `FreEformer/federated/HANDOFF.md` 和实际 `fed_config.py` 为准；需另装对应依赖（包括 `seaborn`）。旧 Flower 入口另需 `flwr`，报表脚本另需 `python-docx`，Notebook 可能需要 Jupyter 和 `openpyxl`。GPU/CUDA、Mamba 和 FDConv 使用各自匹配环境。Windows 的 `.sh` 可在 WSL 中运行。

## 备份前已经存在的运行缺口

这些问题在原样备份时保留，没有顺带改动研究逻辑：

1. 没有统一锁定的依赖环境，文件恢复不等于完整复制旧电脑的软件环境。
2. `ModernTCN.py` 源码已经缺失，而 `Models/modernTCN_running.py` 仍导入它。现存 Python 3.8 字节码保留，但不保证换 Python 版本后可运行。`DLinear` 和 `tcn_baseline` 也有无源码的旧字节码；当前 `dlinear_running.py` 已内置其模型类。
3. `New_Dataset/scripts/rebuild_dataset.py` 的 `WORKSPACE = DATASET_ROOT.parents[2]` 当前会定位到仓库以外；重新构建前需按新电脑位置修正。已有数据 CSV 原样备份，不必重新构建才能取回。
4. `merged_sql.py`、部分 Notebook 和报表脚本含旧电脑绝对路径，换电脑运行前需更新。报表脚本引用的一张旧临时截图已不在当前目录；现有报告文档已保存。

## 清单变更与回退验证

`ORIGINAL_FILE.json` 是备份前已跟踪文件的覆盖清单，`MODIFIED_FILE.json` 是覆盖全部现存研究文件的副本。`DIFF_FILE.patch` 展示新增覆盖内容；`ORIGINAL_HEAD.txt` 保存备份前提交编号，`ORIGINAL_STATUS.txt` 和 `ORIGINAL_UNCOMMITTED.patch` 保存原有未提交变更证据。

`ROLLBACK.sh` 只把原清单复制到明确指定的测试副本，不回滚或删除研究文件，不修改 GitHub 历史；完成后的完整清单保持不变。精确测试命令和结果见 `VERIFICATION.txt`。
