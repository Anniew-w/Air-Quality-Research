# 论文使用指南

## 1. 可直接写入论文的数据描述

本研究数据集整合了 Victoria 现有的 11 个日级 EPA–BOM 配对站点数据，以及 Queensland 政府 2021–2025 年 58 个监测站的小时级空气质量与气象数据。统一后的日级主表包含 100,796 个站点日，覆盖 69 个站点；Queensland 小时级部分包含 2,157,414 条去重后的小时记录。所有原始 Queensland 文件及其元数据、下载 URL 和 SHA-256 校验值均被保留，以支持数据溯源和重复实验。

## 2. 建议的 Methods 写法

### Data sources

Queensland 数据来自州政府 Open Data Portal 的 *Air Quality Monitoring* 2021–2025 年度数据集。该来源按站点发布一小时平均空气质量与气象数据，降雨为一小时累计，时间戳表示平均期开始。现有 Victoria 部分来自 EPA Victoria AirWatch 年度小时均值数据与 Bureau of Meteorology 气象数据形成的既有日级配对数据。

### Harmonisation

所有站点使用稳定的 `location_id`。Queensland 本地时间按 `Australia/Brisbane` 解释。NO2、O3、SO2、NO 和 NOx 从 ppm 转为 ppb；CO 保持 ppm；PM2.5 和 PM10 使用 µg/m3。颗粒物光散射系数统一写入 `bsp`（1/Mm），站点气压单独写入 `barometric_pressure_hpa`，避免与现有字段语义冲突。

### Daily aggregation

污染物和连续气象变量取当日可用小时的算术平均，降雨按小时累计求和，温度和相对湿度计算日最大/最小值，风向使用圆周平均。日太阳曝辐量按 `sum(hourly W/m2) × 0.0036` 转换为 MJ/m2。每个 Queensland 站点日记录 PM2.5 有效小时数与完整率。

### Missing and anomalous data

缺测值保持为空，不作零值替代。负 PM 值作为政府源文件的已发布测量保留，并使用质量字段标记。主分析可要求 `pm25_completeness >= 0.75`，并对不同阈值（例如 0.5、0.9）做敏感性分析。不要在测试集上插值；如训练阶段插值，应仅使用训练时间窗中的信息并报告方法。

### Duplicate resolution

Queensland 原始文件中的 48 个重复小时优先保留非空测量更多的记录。Victoria 既有日级文件中的 6 个跨年边界重复日优先保留 `source_year` 与日期年份一致的记录。所有原始文件均未修改。

## 3. 与澳大利亚标准比较时必须使用正确平均期

2021 年编纂的 National Environment Protection (Ambient Air Quality) Measure 规定的主要标准包括：

| 污染物 | 平均期 | 标准 |
|---|---:|---:|
| CO | 8 小时 | 9.0 ppm |
| NO2 | 1 小时 | 80 ppb |
| NO2 | 1 年 | 15 ppb |
| O3 | 8 小时 | 65 ppb |
| SO2 | 1 小时 | 100 ppb；自 2025-01-01 起 75 ppb |
| SO2 | 1 日 | 20 ppb |
| PM10 | 1 日 | 50 µg/m3 |
| PM10 | 1 年 | 25 µg/m3 |
| PM2.5 | 1 日 | 25 µg/m3 |
| PM2.5 | 1 年 | 8 µg/m3 |

因此：

- 日级表可用于 PM2.5、PM10 和日 SO2 的描述性比较，也可计算年均值。
- CO、O3、小时 NO2 和小时 SO2 的标准不能用日均值代替；必须从小时表计算正确的滚动或小时指标。
- 正式 NEPM 合规评价还涉及监测站资格、数据捕获率、例外事件和辖区报告规则。本数据集的阈值比较只能描述为“超过浓度基准”，不能直接表述为法律上的“不合规”。
- PM 和 O3 的例外事件（如山火、获批计划烧除或大陆尺度扬尘）需要外部事件资料；本数据集未自动分类例外事件。

## 4. 适合的研究问题

1. 多站点 PM2.5 日级或小时级预测，以及跨站点迁移能力。
2. 联邦学习在非独立同分布站点数据上的表现，并比较宏平均与样本加权结果。
3. 气象变量、季节性与 PM2.5/PM10 的时滞关系。
4. 政府运营站点与行业运营站点的分布差异和敏感性分析。
5. 不同站点类别、地理区域或年份的空气污染趋势，但需控制站点启停与监测参数变化。

## 5. 建模与统计建议

- 使用按时间顺序的训练/验证/测试划分；推荐保留最后完整年份或最后 20% 时间作为测试集。
- 至少报告 persistence、seasonal naive 和简单线性/树模型基线，再比较深度模型。
- 预测指标建议同时给出 MAE、RMSE；如使用百分比误差，应说明接近零浓度时的处理。R² 不能单独作为准确性指标。
- 多站点结果应同时报告：每站点指标、站点宏平均、按样本数加权平均，以及站点间四分位数范围。
- 模型选择只能使用训练/验证集，不能根据测试集选超参数或最佳随机种子。
- 对 PM 负值至少做两套分析：原值保留的主数据审计，以及将负值设为缺测或检测限规则下的敏感性分析；不要无说明地截断为零。
- 站点缺测具有结构性：有些站点只监测气象，有些污染物季节性停机。应报告每个目标变量的站点—年份覆盖，而不是仅报告总行数。
- 跨州比较时控制监测方法、站点分类、运营主体与时间粒度差异。Victoria 当前纳入的是既有日级配对数据；Queensland 日级值是本次由小时值派生。

## 6. 论文中应报告的最小信息

- 数据下载/访问日期：2026-08-20。
- 使用的确切站点、年份、目标变量和单位。
- 训练、验证、测试的日期边界和各自样本数。
- 缺测筛选阈值、负值规则、重复值规则和插值方法（如有）。
- 每站点数据完整率、最终纳入/排除站点清单。
- 所有随机种子、软件版本、超参数搜索范围和停止规则。
- 原始数据 URL、许可、数据版本或 SHA-256；逐文件信息见 `metadata/qld_resource_manifest.csv`。
- 山火、扬尘等例外事件是否识别、如何识别，以及是否纳入年均值。

## 7. 主要局限

- 数据集目前扩展到两个州，并非全澳大利亚所有辖区；“Australian stations”应表述为“stations in Victoria and Queensland”。
- Victoria 与 Queensland 的处理路径不同，前者是既有日级 EPA–BOM 配对数据，后者保留小时原始值并派生日均值。
- Queensland 部分包含行业运营站点，政府页面说明某些数据按收到内容发布；需做分层或敏感性分析。
- 仪器、采样方法、站点类别和污染物覆盖随站点和年份变化，空白不表示零浓度。
- EPA Victoria 指出实时/历史数据可能因设备、供电、维护或后续验证而调整，且部分近道路站点可能不满足全部选址距离要求。
- Bureau of Meteorology 派生气象字段的再分发和商业使用需遵守具体产品条款并正确署名。

## 8. References

完整可点击 URL 和访问日期见 `metadata/source_references.csv`。论文参考文献至少应包括：

1. Queensland Government. *Air Quality Monitoring* datasets, 2021–2025.
2. Environment Protection Authority Victoria. *EPA Air Watch All Sites Air Quality Hourly Averages – Yearly*.
3. Bureau of Meteorology. *Climate Data Online*.
4. Federal Register of Legislation. *National Environment Protection (Ambient Air Quality) Measure*, compilation dated 18 May 2021.
5. National Environment Protection Council. *Variation to Ambient Air Quality NEPM – ozone, nitrogen dioxide and sulfur dioxide*.
6. Australian Government Department of Health and Aged Care. *PM2.5 air quality categories and public health advice*.

