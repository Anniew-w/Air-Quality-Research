#!/usr/bin/env python3
"""Rebuild the harmonised Australian air-quality dataset.

The script has two subcommands:
  prepare-downloads  Build a curl config from Queensland CKAN metadata.
  build              Harmonise downloaded Queensland hourly CSV files and
                     the existing Victorian daily Finalized_Dataset files.

It intentionally preserves raw measurements, including negative particulate
values published by the source. Quality flags and completeness fields are
added; measurements are not silently clipped or imputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd


DATASET_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = DATASET_ROOT.parents[2]
SOURCE_FINALIZED = WORKSPACE / "Finalized_Dataset"
TMP_METADATA = Path("/private/tmp/air-quality-new-dataset")
RAW_ROOT = DATASET_ROOT / "raw/qld"
RAW_METADATA = RAW_ROOT / "metadata"
RAW_SITE_YEAR = RAW_ROOT / "site_year"
DAILY_DIR = DATASET_ROOT / "daily/by_station"
HOURLY_DIR = DATASET_ROOT / "hourly/by_station"
META_DIR = DATASET_ROOT / "metadata"

YEARS = tuple(range(2021, 2026))
QLD_DATASET_URL = "https://www.data.qld.gov.au/dataset/air-quality-monitoring-{year}"
QLD_CKAN_URL = (
    "https://www.data.qld.gov.au/api/3/action/package_show?"
    "id=air-quality-monitoring-{year}"
)


DAILY_COLUMNS = [
    "Date", "location_id", "bsp", "co", "dbt", "no2", "o3", "pm10",
    "sigma05", "so2", "swd", "sws", "vwd", "vws", "location_name",
    "source_year", "sigma60", "station", "evapotranspiration", "rain",
    "pan_evaporation", "max_temperature", "min_temperature",
    "max_rel_humidity", "min_rel_humidity", "avg_10m_wind_speed",
    "solar_radiation", "pm25", "state", "country", "latitude",
    "longitude", "timezone", "frequency", "valid_hour_count",
    "expected_hour_count", "pm25_valid_hours", "pm25_completeness",
    "negative_pm25_hours", "negative_pm10_hours", "site_classification",
    "industry_operated", "source_name", "source_url", "source_dataset_url",
    "visibility_reducing_particles", "total_suspended_particles", "benzene",
    "toluene", "ethylbenzene", "xylene", "formaldehyde",
    "hydrogen_sulfide", "total_reduced_sulfur",
    "solar_radiation_w_m2_mean", "relative_humidity_mean",
    "barometric_pressure_hpa", "no", "nox", "wind_speed_std_dev",
    "carbon_dioxide", "methane",
]

HOURLY_COLUMNS = [
    "Date", "DateTime", "Time", "location_id", "bsp", "co", "dbt", "no2",
    "o3", "pm10", "sigma05", "so2", "swd", "sws", "vwd", "vws",
    "location_name", "source_year", "sigma60", "station", "rain", "pm25",
    "state", "country", "latitude", "longitude", "timezone", "frequency",
    "relative_humidity", "solar_radiation_w_m2",
    "visibility_reducing_particles", "total_suspended_particles", "benzene",
    "toluene", "ethylbenzene", "xylene", "formaldehyde",
    "hydrogen_sulfide", "total_reduced_sulfur", "barometric_pressure_hpa",
    "no", "nox", "wind_speed_std_dev", "carbon_dioxide", "methane",
    "site_classification",
    "industry_operated", "source_name", "source_url", "source_dataset_url",
    "quality_flag",
]


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def slugify(value: str) -> str:
    text = value.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown_station"


def station_label(resource_name: str, year: int) -> str:
    return re.sub(rf"\s+{year}\s+hourly.*$", "", resource_name, flags=re.I).strip()


def station_short_name(label: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()


def parse_lat_lon(description: str) -> tuple[float | None, float | None]:
    match = re.search(
        r"Latitude:\s*([-+]?\d+(?:\.\d+)?)\s*;\s*Longitude:\s*([-+]?\d+(?:\.\d+)?)",
        description,
        flags=re.I,
    )
    return (float(match.group(1)), float(match.group(2))) if match else (None, None)


def parse_classification(description: str) -> str:
    match = re.search(
        r"AS/NZS\s+3580\.1\.1\s+site classification:\*\*\s*([^\r\n]+)",
        description,
        flags=re.I,
    )
    return clean_space(match.group(1)).strip("* ") if match else ""


def url_filename(url: str, resource_id: str) -> str:
    name = Path(urlparse(url).path).name or f"{resource_id}.csv"
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return f"{resource_id}__{name}"


def load_metadata_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resources: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for year in YEARS:
        source_path = TMP_METADATA / f"qld-{year}.json"
        if not source_path.exists():
            source_path = RAW_METADATA / f"qld-{year}.json"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing CKAN metadata: {source_path}")
        RAW_METADATA.mkdir(parents=True, exist_ok=True)
        copied_path = RAW_METADATA / source_path.name
        if source_path.resolve() != copied_path.resolve():
            shutil.copy2(source_path, copied_path)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        result = payload["result"]
        datasets.append({
            "year": year,
            "dataset_id": result.get("id", ""),
            "dataset_name": result.get("name", ""),
            "dataset_title": result.get("title", ""),
            "publisher": result.get("organization", {}).get("title", ""),
            "license": result.get("license_title", ""),
            "license_url": result.get("license_url", ""),
            "data_last_updated": result.get("data_last_updated", ""),
            "metadata_modified": result.get("metadata_modified", ""),
            "dataset_url": QLD_DATASET_URL.format(year=year),
            "ckan_url": QLD_CKAN_URL.format(year=year),
            "resolution": "1 hour average; rainfall is a 1 hour sum",
            "timestamp_definition": "start of averaging period",
            "timezone": "Australia/Brisbane (AEST, UTC+10; no daylight saving)",
        })
        for resource in result.get("resources", []):
            url = clean_space(resource.get("url"))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            description = str(resource.get("description") or "")
            lat, lon = parse_lat_lon(description)
            label = station_label(str(resource.get("name") or ""), year)
            short = station_short_name(label)
            station_slug = slugify(short)
            resource_id = str(resource.get("id") or hashlib.sha1(url.encode()).hexdigest()[:12])
            output_path = RAW_SITE_YEAR / str(year) / url_filename(url, resource_id)
            resources.append({
                "year": year,
                "resource_id": resource_id,
                "resource_name": clean_space(resource.get("name")),
                "station_label": label,
                "location_name": short,
                "station_slug": station_slug,
                "location_id": f"QLD_{station_slug.upper()}",
                "latitude": lat,
                "longitude": lon,
                "site_classification": parse_classification(description),
                "industry_operated": "Industry-operated" in description,
                "source_url": url,
                "source_dataset_url": QLD_DATASET_URL.format(year=year),
                "license": result.get("license_title", ""),
                "description": clean_space(description),
                "local_raw_path": str(output_path.relative_to(DATASET_ROOT)),
            })
    return resources, datasets


def prepare_downloads() -> None:
    resources, datasets = load_metadata_records()
    RAW_SITE_YEAR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    config_path = TMP_METADATA / "qld-downloads.curl"
    pending = [
        row for row in resources
        if not (DATASET_ROOT / row["local_raw_path"]).exists()
        or (DATASET_ROOT / row["local_raw_path"]).stat().st_size == 0
    ]
    with config_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(pending):
            output_path = DATASET_ROOT / row["local_raw_path"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            handle.write(f'url = "{row["source_url"]}"\n')
            handle.write(f'output = "{output_path}"\n')
            handle.write("location\nfail\nsilent\nshow-error\ncreate-dirs\n")
            if index < len(pending) - 1:
                handle.write("next\n")
    pd.DataFrame(resources).to_csv(META_DIR / "qld_resource_manifest.csv", index=False)
    pd.DataFrame(datasets).to_csv(META_DIR / "qld_dataset_manifest.csv", index=False)
    print(json.dumps({
        "curl_config": str(config_path),
        "unique_resources": len(resources),
        "pending_resources": len(pending),
        "datasets": len(datasets),
        "output_root": str(RAW_SITE_YEAR),
    }, indent=2))


def norm_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", header.lower()).strip()


def canonical_for_header(header: str) -> str | None:
    h = norm_header(header)
    if h == "date":
        return "Date"
    if h == "time":
        return "Time"
    patterns = [
        (r"pm\s*2\s*5|pm25", "pm25"),
        (r"pm\s*10|pm10", "pm10"),
        (r"nitrogen dioxide|\bno2\b", "no2"),
        (r"nitrogen oxides|\bnox\b", "nox"),
        (r"nitrogen oxide|nitric oxide|\bno\b", "no"),
        (r"\bozone\b|\bo3\b", "o3"),
        (r"sulfur dioxide|sulphur dioxide|\bso2\b", "so2"),
        (r"carbon monoxide|\bco\b", "co"),
        (r"barometric pressure|air pressure", "barometric_pressure_hpa"),
        (r"temperature", "dbt"),
        (r"relative humidity", "relative_humidity"),
        (r"wind sigma theta|sigma theta", "sigma60"),
        (r"wind speed std dev|wind speed standard deviation", "wind_speed_std_dev"),
        (r"wind direction", "swd"),
        (r"wind speed", "sws"),
        (r"rainfall|precipitation", "rain"),
        (r"solar radiation", "solar_radiation_w_m2"),
        (r"visibility reducing particles|nephel", "visibility_reducing_particles"),
        (r"total suspended particles|\btsp\b", "total_suspended_particles"),
        (r"\bbenzene\b", "benzene"),
        (r"\btoluene\b", "toluene"),
        (r"ethylbenzene", "ethylbenzene"),
        (r"xylene", "xylene"),
        (r"formaldehyde", "formaldehyde"),
        (r"hydrogen sulfide|hydrogen sulphide|\bh2s\b", "hydrogen_sulfide"),
        (r"total reduced sulfur|total reduced sulphur|\btrs\b", "total_reduced_sulfur"),
        (r"carbon dioxide|\bco2\b", "carbon_dioxide"),
        (r"\bmethane\b|\bch4\b", "methane"),
    ]
    for pattern, canonical in patterns:
        if re.search(pattern, h):
            return canonical
    return None


def numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().replace({"": np.nan, "-": np.nan, "NA": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def add_missing_and_order(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[columns]


def first_valid(series: pd.Series) -> Any:
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else np.nan


def circular_mean_degrees(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return np.nan
    radians = np.deg2rad(vals)
    angle = math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean()))
    return angle % 360.0


def harmonise_qld_hourly(
    raw: pd.DataFrame, meta: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, str], int]:
    rename: dict[str, str] = {}
    mapped_originals: dict[str, str] = {}
    for original in raw.columns:
        canonical = canonical_for_header(str(original))
        if canonical and canonical not in rename.values():
            rename[original] = canonical
            mapped_originals[canonical] = str(original)
    data = raw.rename(columns=rename).copy()
    if "Date" not in data.columns or "Time" not in data.columns:
        raise ValueError("Required Date/Time columns were not found")
    dt = pd.to_datetime(
        data["Date"].astype(str).str.strip() + " " + data["Time"].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
    )
    data = data.loc[dt.notna()].copy()
    dt = dt.loc[dt.notna()]
    data["DateTime"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    data["Date"] = dt.dt.strftime("%Y-%m-%d")
    data["Time"] = dt.dt.strftime("%H:%M:%S")
    numeric_cols = [
        "bsp", "co", "dbt", "no2", "no", "nox", "o3", "pm10", "so2", "swd", "sws",
        "rain", "pm25", "relative_humidity", "solar_radiation_w_m2",
        "visibility_reducing_particles", "total_suspended_particles", "benzene",
        "toluene", "ethylbenzene", "xylene", "formaldehyde",
        "hydrogen_sulfide", "total_reduced_sulfur", "barometric_pressure_hpa",
        "sigma60", "wind_speed_std_dev", "carbon_dioxide", "methane",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = numeric(data[col])
    # The existing Victorian NO2/O3/SO2 fields are in ppb. Queensland source
    # files publish these gases in ppm, so convert to ppb for compatibility.
    for col in ("no2", "no", "nox", "o3", "so2"):
        original = mapped_originals.get(col, "")
        if col in data.columns and re.search(r"\bppm\b", original, flags=re.I):
            data[col] = data[col] * 1000.0
    if "visibility_reducing_particles" in data.columns:
        # EPA Victoria defines BSP as particle light scattering in 1/Mm.
        data["bsp"] = data["visibility_reducing_particles"]
    data["location_id"] = meta["location_id"]
    data["location_name"] = meta["location_name"]
    data["source_year"] = int(meta["year"])
    data["station"] = meta["location_name"]
    data["state"] = "Queensland"
    data["country"] = "Australia"
    data["latitude"] = meta.get("latitude")
    data["longitude"] = meta.get("longitude")
    data["timezone"] = "Australia/Brisbane"
    data["frequency"] = "hourly"
    data["site_classification"] = meta.get("site_classification", "")
    data["industry_operated"] = bool(meta.get("industry_operated", False))
    data["source_name"] = "Queensland Government Air Quality Monitoring"
    data["source_url"] = meta["source_url"]
    data["source_dataset_url"] = meta["source_dataset_url"]
    neg_pm25 = data.get("pm25", pd.Series(np.nan, index=data.index)).lt(0)
    neg_pm10 = data.get("pm10", pd.Series(np.nan, index=data.index)).lt(0)
    data["quality_flag"] = np.select(
        [neg_pm25 & neg_pm10, neg_pm25, neg_pm10],
        ["negative_pm25_and_pm10_as_published", "negative_pm25_as_published", "negative_pm10_as_published"],
        default="",
    )
    source_duplicate_timestamps = int(data.duplicated(["DateTime"]).sum())
    if source_duplicate_timestamps:
        measurement_cols = [col for col in numeric_cols if col in data.columns]
        data["_non_null_measurements"] = data[measurement_cols].notna().sum(axis=1)
        data = data.sort_values(
            ["DateTime", "_non_null_measurements"],
            ascending=[True, False],
            kind="stable",
        )
        data = data.drop_duplicates(subset=["DateTime"], keep="first")
        data = data.sort_values("DateTime", kind="stable").drop(columns=["_non_null_measurements"])
    return add_missing_and_order(data, HOURLY_COLUMNS), mapped_originals, source_duplicate_timestamps


def qld_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    numeric_mean_cols = [
        "bsp", "co", "dbt", "no2", "no", "nox", "o3", "pm10", "sigma05", "sigma60", "so2", "sws",
        "pm25", "visibility_reducing_particles", "total_suspended_particles",
        "benzene", "toluene", "ethylbenzene", "xylene", "formaldehyde",
        "hydrogen_sulfide", "total_reduced_sulfur", "relative_humidity",
        "solar_radiation_w_m2", "barometric_pressure_hpa", "wind_speed_std_dev",
        "carbon_dioxide", "methane",
    ]
    rows: list[dict[str, Any]] = []
    for date, group in hourly.groupby("Date", sort=True):
        row: dict[str, Any] = {"Date": date}
        for col in numeric_mean_cols:
            row[col] = pd.to_numeric(group[col], errors="coerce").mean()
        row["swd"] = circular_mean_degrees(group["swd"])
        row["rain"] = pd.to_numeric(group["rain"], errors="coerce").sum(min_count=1)
        row["max_temperature"] = pd.to_numeric(group["dbt"], errors="coerce").max()
        row["min_temperature"] = pd.to_numeric(group["dbt"], errors="coerce").min()
        row["max_rel_humidity"] = pd.to_numeric(group["relative_humidity"], errors="coerce").max()
        row["min_rel_humidity"] = pd.to_numeric(group["relative_humidity"], errors="coerce").min()
        row["avg_10m_wind_speed"] = pd.to_numeric(group["sws"], errors="coerce").mean()
        solar = pd.to_numeric(group["solar_radiation_w_m2"], errors="coerce")
        row["solar_radiation"] = solar.sum(min_count=1) * 0.0036
        row["solar_radiation_w_m2_mean"] = solar.mean()
        row["relative_humidity_mean"] = pd.to_numeric(group["relative_humidity"], errors="coerce").mean()
        row["valid_hour_count"] = int(group["DateTime"].nunique())
        row["expected_hour_count"] = 24
        row["pm25_valid_hours"] = int(pd.to_numeric(group["pm25"], errors="coerce").notna().sum())
        row["pm25_completeness"] = row["pm25_valid_hours"] / 24.0
        row["negative_pm25_hours"] = int(pd.to_numeric(group["pm25"], errors="coerce").lt(0).sum())
        row["negative_pm10_hours"] = int(pd.to_numeric(group["pm10"], errors="coerce").lt(0).sum())
        for col in [
            "location_id", "location_name", "source_year", "station", "state", "country",
            "latitude", "longitude", "timezone", "site_classification", "industry_operated",
            "source_name", "source_url", "source_dataset_url",
        ]:
            row[col] = first_valid(group[col])
        row["frequency"] = "daily_from_hourly"
        rows.append(row)
    return add_missing_and_order(pd.DataFrame(rows), DAILY_COLUMNS)


def standardise_existing_vic(path: Path) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(path, low_memory=False)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["Date"]).copy()
    frame["state"] = "Victoria"
    frame["country"] = "Australia"
    frame["timezone"] = "Australia/Melbourne"
    frame["frequency"] = "daily_existing_finalized"
    frame["source_name"] = "EPA Victoria AirWatch + Bureau of Meteorology (existing finalized data)"
    frame["source_url"] = "https://discover.data.vic.gov.au/en_AU/dataset/epa-air-watch-all-sites-air-quality-hourly-averages-yearly"
    frame["source_dataset_url"] = "https://www.bom.gov.au/climate/data/index.shtml"
    frame["state"] = "Victoria"
    frame["industry_operated"] = False
    source_duplicate_timestamps = int(frame.duplicated(["Date"]).sum())
    if source_duplicate_timestamps:
        date_year = pd.to_datetime(frame["Date"], errors="coerce").dt.year
        source_year = pd.to_numeric(frame["source_year"], errors="coerce")
        frame["_year_match"] = source_year.eq(date_year)
        frame["_non_null_measurements"] = frame.notna().sum(axis=1)
        frame = frame.sort_values(
            ["Date", "_year_match", "_non_null_measurements"],
            ascending=[True, False, False],
            kind="stable",
        )
        frame = frame.drop_duplicates(subset=["Date"], keep="first")
        frame = frame.sort_values("Date", kind="stable").drop(
            columns=["_year_match", "_non_null_measurements"]
        )
    return add_missing_and_order(frame, DAILY_COLUMNS), source_duplicate_timestamps


def read_resource_manifest() -> pd.DataFrame:
    path = META_DIR / "qld_resource_manifest.csv"
    if not path.exists():
        raise FileNotFoundError("Run prepare-downloads first")
    frame = pd.read_csv(path)
    frame["resource_id"] = frame["resource_id"].astype(str)
    return frame


def build_dataset() -> None:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (DAILY_DIR, HOURLY_DIR):
        for old in folder.glob("*.csv"):
            old.unlink()

    daily_frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    original_column_records: list[dict[str, str]] = []
    qld_hourly_rows_total = 0

    for path in sorted(SOURCE_FINALIZED.glob("*.csv")):
        frame, source_duplicates = standardise_existing_vic(path)
        output = DAILY_DIR / path.name
        frame.to_csv(output, index=False, date_format="%Y-%m-%d", quoting=csv.QUOTE_MINIMAL)
        daily_frames.append(frame)
        quality_rows.append({
            "location_id": str(first_valid(frame["location_id"])),
            "location_name": first_valid(frame["location_name"]),
            "state": "Victoria",
            "frequency": "daily",
            "start": frame["Date"].min(),
            "end": frame["Date"].max(),
            "rows": len(frame),
            "unique_timestamps": frame["Date"].nunique(),
            "duplicate_timestamps": int(frame.duplicated(["Date"]).sum()),
            "source_duplicate_timestamps_removed": source_duplicates,
            "pm25_non_null": int(pd.to_numeric(frame["pm25"], errors="coerce").notna().sum()),
            "pm25_missing_pct": float(pd.to_numeric(frame["pm25"], errors="coerce").isna().mean()),
            "negative_pm25_count": int(pd.to_numeric(frame["pm25"], errors="coerce").lt(0).sum()),
            "source": "existing Finalized_Dataset",
        })

    manifest = read_resource_manifest()
    downloaded_bytes: list[int] = []
    downloaded_sha256: list[str] = []
    download_status: list[str] = []
    for _, meta_row in manifest.iterrows():
        raw_path = DATASET_ROOT / str(meta_row["local_raw_path"])
        if raw_path.exists() and raw_path.stat().st_size > 0:
            downloaded_bytes.append(raw_path.stat().st_size)
            digest = hashlib.sha256()
            with raw_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            downloaded_sha256.append(digest.hexdigest())
            download_status.append("downloaded")
        else:
            downloaded_bytes.append(0)
            downloaded_sha256.append("")
            download_status.append("missing")
    manifest["downloaded_bytes"] = downloaded_bytes
    manifest["sha256"] = downloaded_sha256
    manifest["download_status"] = download_status
    manifest["retrieved_date"] = "2026-08-20"
    manifest.to_csv(META_DIR / "qld_resource_manifest.csv", index=False)
    for _, meta_row in manifest.sort_values(["year", "station_slug", "resource_id"]).iterrows():
        meta = meta_row.where(pd.notna(meta_row), None).to_dict()
        raw_path = DATASET_ROOT / str(meta["local_raw_path"])
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing downloaded source file: {raw_path}")
        raw = pd.read_csv(raw_path, dtype=str, low_memory=False, encoding="utf-8-sig")
        hourly, mappings, source_duplicates = harmonise_qld_hourly(raw, meta)
        if hourly.empty:
            continue
        qld_hourly_rows_total += len(hourly)
        station_slug = str(meta["station_slug"])
        station_output = HOURLY_DIR / f"qld_{station_slug}_hourly.csv"
        hourly.to_csv(
            station_output,
            mode="a",
            header=not station_output.exists(),
            index=False,
            date_format="%Y-%m-%d",
            quoting=csv.QUOTE_MINIMAL,
        )
        daily = qld_daily(hourly)
        daily_frames.append(daily)
        for canonical, original in mappings.items():
            original_column_records.append({
                "year": str(meta["year"]),
                "location_id": str(meta["location_id"]),
                "location_name": str(meta["location_name"]),
                "source_column": original,
                "canonical_column": canonical,
                "source_url": str(meta["source_url"]),
            })
        quality_rows.append({
            "location_id": str(meta["location_id"]),
            "location_name": str(meta["location_name"]),
            "state": "Queensland",
            "frequency": "hourly",
            "start": hourly["DateTime"].min(),
            "end": hourly["DateTime"].max(),
            "rows": len(hourly),
            "unique_timestamps": hourly["DateTime"].nunique(),
            "duplicate_timestamps": int(hourly.duplicated(["DateTime"]).sum()),
            "source_duplicate_timestamps_removed": source_duplicates,
            "pm25_non_null": int(pd.to_numeric(hourly["pm25"], errors="coerce").notna().sum()),
            "pm25_missing_pct": float(pd.to_numeric(hourly["pm25"], errors="coerce").isna().mean()),
            "negative_pm25_count": int(pd.to_numeric(hourly["pm25"], errors="coerce").lt(0).sum()),
            "source": str(meta["source_url"]),
        })

    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_daily = add_missing_and_order(all_daily, DAILY_COLUMNS)
    all_daily = all_daily.sort_values(["state", "location_id", "Date"], kind="stable")
    all_daily.to_csv(DATASET_ROOT / "daily/all_stations_daily.csv", index=False)

    qld_daily_all = all_daily.loc[all_daily["state"].eq("Queensland")].copy()
    for location_id, group in qld_daily_all.groupby("location_id", sort=True):
        station_slug = slugify(str(first_valid(group["location_name"])))
        group.sort_values("Date").to_csv(
            DAILY_DIR / f"qld_{station_slug}_merged.csv", index=False
        )

    quality = pd.DataFrame(quality_rows)
    quality.to_csv(META_DIR / "quality_by_resource.csv", index=False)
    if original_column_records:
        pd.DataFrame(original_column_records).drop_duplicates().to_csv(
            META_DIR / "qld_column_mapping_observed.csv", index=False
        )

    station_rows: list[dict[str, Any]] = []
    for (state, location_id), group in all_daily.groupby(["state", "location_id"], dropna=False):
        station_rows.append({
            "location_id": location_id,
            "location_name": first_valid(group["location_name"]),
            "state": state,
            "country": first_valid(group["country"]),
            "latitude": first_valid(group["latitude"]),
            "longitude": first_valid(group["longitude"]),
            "timezone": first_valid(group["timezone"]),
            "site_classification": first_valid(group["site_classification"]),
            "industry_operated": first_valid(group["industry_operated"]),
            "daily_start": group["Date"].min(),
            "daily_end": group["Date"].max(),
            "daily_rows": len(group),
            "years_present": ",".join(map(str, sorted(pd.to_numeric(group["source_year"], errors="coerce").dropna().astype(int).unique()))),
            "pm25_daily_non_null": int(pd.to_numeric(group["pm25"], errors="coerce").notna().sum()),
            "pm10_daily_non_null": int(pd.to_numeric(group["pm10"], errors="coerce").notna().sum()),
            "no2_daily_non_null": int(pd.to_numeric(group["no2"], errors="coerce").notna().sum()),
            "o3_daily_non_null": int(pd.to_numeric(group["o3"], errors="coerce").notna().sum()),
            "so2_daily_non_null": int(pd.to_numeric(group["so2"], errors="coerce").notna().sum()),
            "co_daily_non_null": int(pd.to_numeric(group["co"], errors="coerce").notna().sum()),
        })
    stations = pd.DataFrame(station_rows).sort_values(["state", "location_name"])
    stations.to_csv(META_DIR / "stations.csv", index=False)

    summary = {
        "dataset_version": "2026-08-20",
        "daily_rows": int(len(all_daily)),
        "stations_total": int(len(stations)),
        "stations_by_state": stations.groupby("state").size().astype(int).to_dict(),
        "daily_date_start": str(all_daily["Date"].min()),
        "daily_date_end": str(all_daily["Date"].max()),
        "qld_hourly_files": int(len(list(HOURLY_DIR.glob("*.csv")))),
        "qld_raw_resource_files": int(len(list(RAW_SITE_YEAR.glob("*/*.csv")))),
        "qld_hourly_rows": int(qld_hourly_rows_total),
        "source_victoria_daily_files": int(len(list(SOURCE_FINALIZED.glob("*.csv")))),
        "qld_source_years": list(YEARS),
    }
    (META_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def write_output_manifest() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(DATASET_ROOT.rglob("*")):
        if not path.is_file() or path.name == "output_file_manifest.csv":
            continue
        relative = path.relative_to(DATASET_ROOT)
        if "node_modules" in relative.parts or path.name.endswith(".inspect.ndjson"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append({
            "relative_path": str(relative),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "category": relative.parts[0] if relative.parts else "root",
        })
    pd.DataFrame(rows).to_csv(META_DIR / "output_file_manifest.csv", index=False)
    print(json.dumps({"manifested_files": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare-downloads", "build", "manifest"])
    args = parser.parse_args()
    if args.command == "prepare-downloads":
        prepare_downloads()
    elif args.command == "build":
        build_dataset()
    else:
        write_output_manifest()


if __name__ == "__main__":
    main()
