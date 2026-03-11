import pandas as pd
from pathlib import Path

# === 路径设置 ===
base = Path("/Users/wangshihan/Air Quality Research")
epa_dir = base / "Merged_EPA"
bom_dir = base / "Cleaned_BOM_Merged"
map_path = base / "epa_to_bom.csv"
out_dir = base / "EPA_BOM_Merged"
out_dir.mkdir(exist_ok=True)

# === 读取映射表 ===
mapping = pd.read_csv(map_path)
print(f"🔗 读取映射表，共 {len(mapping)} 条匹配")

def clean_name(name: str):
    """统一命名：去空格、小写、替换括号和横线"""
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

# === 循环每对匹配关系 ===
for _, row in mapping.iterrows():
    epa_name = clean_name(row["location_name"])
    bom_name = clean_name(row["bom_name"])
    loc_id = row["location_id"]

    # 寻找文件
    epa_file = list(epa_dir.glob(f"{row['location_name'].replace(' ', '_')}_{loc_id}.csv"))
    bom_file = list(bom_dir.glob(f"{bom_name}*.csv"))

    if not epa_file:
        print(f"⚠️ 未找到 EPA 文件: {epa_name}")
        continue
    if not bom_file:
        print(f"⚠️ 未找到 BOM 文件: {bom_name}")
        continue

    try:
        # === 读取数据 ===
        epa_df = pd.read_csv(epa_file[0])
        bom_df = pd.read_csv(bom_file[0])

        # 找日期列
        epa_date_col = [c for c in epa_df.columns if "date" in c.lower()][0]
        bom_date_col = [c for c in bom_df.columns if "date" in c.lower()][0]

        # 标准化日期格式
        epa_df[epa_date_col] = pd.to_datetime(epa_df[epa_date_col], errors="coerce")
        bom_df[bom_date_col] = pd.to_datetime(bom_df[bom_date_col], errors="coerce")

        # 合并
        merged = pd.merge(epa_df, bom_df, left_on=epa_date_col, right_on=bom_date_col, how="inner")

        # 保存结果
        out_path = out_dir / f"{row['location_name']}_{loc_id}_merged.csv"
        merged.to_csv(out_path, index=False)
        print(f"✅ 合并成功: {row['location_name']} ({loc_id}) -> {len(merged)} 行")

    except Exception as e:
        print(f"❌ 合并失败: {row['location_name']} ({loc_id}) - {e}")

print("\n🎯 所有匹配站点处理完成！")