# New Dataset：澳大利亚空气质量与气象数据集

版本：2026-08-20  
覆盖：2021-01-01 至 2025-12-31  
地区：Victoria、Queensland

## 数据集概况

- 日级主表：100,796 行、69 个站点，其中 Victoria 11 站、Queensland 58 站。
- 小时级表：2,157,414 行、58 个 Queensland 站点，按站点拆分为 58 个 CSV。
- 原始网页数据：251 个 Queensland 政府站点—年份 CSV，均保留原文件和 SHA-256 校验值。
- 时间扩展：现有 Victoria 数据为 2021–2024；新增 Queensland 数据为 2021–2025，因此总体扩展至 2025 年。
- 新增属性包括：站点经纬度、州、时区、站点分类、行业运营标志、气压、相对湿度、NO、NOx、风向离散度、风速标准差、TSP、部分空气毒物、CO2、CH4、有效小时数及 PM2.5 完整率。

## 目录结构

```text
New_Dataset/
├── daily/
│   ├── all_stations_daily.csv        # 推荐的日级统一主表
│   └── by_station/                    # 69 个 Finalized-style 站点文件
├── hourly/
│   └── by_station/                    # 58 个 Queensland 小时级站点文件
├── raw/qld/
│   ├── metadata/                      # CKAN 年度数据集元数据 JSON
│   └── site_year/                     # 251 个未经改写的官方 CSV
├── metadata/
│   ├── stations.csv                   # 站点目录和污染物覆盖情况
│   ├── data_dictionary.csv            # 字段、单位和处理规则
│   ├── source_references.csv           # 数据、标准和健康资料引用
│   ├── qld_dataset_manifest.csv        # 年度数据集级来源清单
│   ├── qld_resource_manifest.csv       # 逐文件 URL、许可、大小和 SHA-256
│   ├── qld_column_mapping_observed.csv # 原始列名到统一列名的实际映射
│   ├── quality_by_resource.csv         # 逐来源质量统计
│   └── dataset_summary.json            # 机器可读总览
├── PAPER_GUIDE.md                      # 论文方法、标准、局限与引用建议
├── New_Dataset_catalog.xlsx            # 可视化目录和质量审阅工作簿
└── scripts/rebuild_dataset.py           # 可复现的数据构建脚本
```

## 合并与清洗规则

1. 现有 `Finalized_Dataset` 的 11 个 Victoria 日级文件作为既有研究数据纳入，不改写原文件；新目录中生成统一列顺序的副本。
2. Queensland 政府 2021–2025 年站点级小时 CSV 通过年度 CKAN 元数据清单下载。完全相同的资源 URL 在下载前去重。
3. Queensland 时间戳为当地 AEST，数据行表示一小时平均期的开始；`Australia/Brisbane` 全年无夏令时。
4. `NO2`、`O3`、`SO2`、`NO` 和 `NOx` 的 Queensland 原始单位为 ppm；为匹配现有 Victoria 字段，统一乘以 1000 写成 ppb。`CO` 保持 ppm。
5. `bsp` 在现有 Victoria 数据中是颗粒物光散射系数（1/Mm），不是气压。Queensland 的 `Visibility-reducing Particles` 映射到 `bsp`；气压独立写入 `barometric_pressure_hpa`。
6. 日级污染物、气压、温度、湿度和风速采用可用小时的算术平均；日最高/最低温湿度取小时极值；降雨是小时和；风向采用圆周平均。
7. Queensland 日太阳辐射由小时均值积分为日曝辐量：`sum(W/m2) × 0.0036 = MJ/m2`。
8. Lytton 和 Wynnum North 的 2021 原始文件各含 24 个重复小时。清洗副本优先保留非空测量字段更多的记录，共删除 48 个重复小时；原始文件不变。
9. 现有 Victoria 日表存在 6 个跨年边界重复日。清洗副本优先保留 `source_year` 与日期年份一致的记录；原始文件不变。
10. 缺测保持为空，不把“未监测”当成零。官方发布的负 PM2.5/PM10 值保留，并写入 `quality_flag` 或日级负值小时计数。

## 推荐使用方式

- 日级预测：使用 `daily/all_stations_daily.csv`，按 `location_id` 分组并按日期排序。
- 小时级预测：从 `hourly/by_station/` 选择目标站点；不要把所有站点盲目拼成一个没有站点标识的序列。
- PM2.5 日级分析：对 Queensland 可用 `pm25_completeness >= 0.75` 作为明确、可复核的研究筛选条件。该阈值是研究数据完整性规则，不等同于正式 NEPM 合规判定。
- 跨站或联邦学习：报告每个客户端的样本数、缺测率、州、站点类别和运营主体；同时报告站点宏平均与按样本数加权结果。
- 所有时间切分必须按时间先后进行，不能随机打散后再划分训练、验证和测试集。

## 数据质量摘要

- 清洗后日级 `(state, location_id, Date)` 重复数：0。
- 清洗后小时级每站 `DateTime` 重复数：0；无无法解析的时间戳。
- Queensland 小时 PM2.5 非空记录：1,565,668；官方发布的负 PM2.5 小时值：17,043。
- Queensland 有 65,163 个站点日达到至少 18/24 个 PM2.5 有效小时。
- 9 个 Queensland 站点在元数据中被标识为行业运营；使用时应与政府运营站点分层或做敏感性分析。

## 引用与许可

逐数据集引用见 `metadata/source_references.csv`，逐原始文件 URL、许可和 SHA-256 见 `metadata/qld_resource_manifest.csv`。

Queensland 新增数据的许可为 Creative Commons Attribution 4.0。现有 Victoria 空气质量来源页面标注 CC BY 4.0 International。Bureau of Meteorology 派生气象字段应遵守具体产品许可和署名要求；在重新分发或商业使用前，请检查 Bureau 最新版权条款，不能仅凭本数据集说明假定全部 BOM 内容均可按 CC BY 再分发。

建议数据引用写法：

> Queensland Government, *Air Quality Monitoring, 2021–2025*, hourly site-level air-quality and meteorological data, retrieved 20 August 2026, CC BY 4.0; Environment Protection Authority Victoria, *EPA Air Watch All Sites Air Quality Hourly Averages – Yearly*; Bureau of Meteorology, *Climate Data Online*.

## 可复现构建

`scripts/rebuild_dataset.py` 记录了字段映射、单位换算、去重和日聚合规则。已下载原始数据和 CKAN 元数据包含在 `raw/qld/` 中；重建脚本会为每个原始文件重新计算 SHA-256。脚本默认要求工作区同级存在原始 `Finalized_Dataset/`，并使用捆绑环境中的 pandas/numpy。

