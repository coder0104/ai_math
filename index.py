# wifi_floorplan_heatmap_korean.py
# 같은 폴더에 다음 2개를 넣고 실행:
# 1) 인수_data.zip
# 2) 4층 지도 이미지: KakaoTalk_Photo_2026-06-15-20-26-23.jpeg
#
# 필요 패키지:
# pip install pandas numpy matplotlib

import os
import re
import io
import csv
import zipfile
import unicodedata
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =========================
# 0. 기본 설정
# =========================

ZIP_PATH = "인수_data.zip"
FLOORPLAN_IMAGE = "KakaoTalk_Photo_2026-06-15-20-26-23.jpeg"

OUT_DIR = Path("wifi_floorplan_outputs")
OUT_DIR.mkdir(exist_ok=True)

ROOMS = list(range(401, 416))
POINT_ORDER = ["공유기", "문앞", "벽", "복도"]

POINT_ALIASES = {
    "공유기": "공유기",
    "AP": "공유기",
    "ap": "공유기",
    "책상": "공유기",

    "문앞": "문앞",
    "문쪽": "문앞",
    "문": "문앞",
    "소파쪽문": "문앞",
    "소파쪽 문": "문앞",

    "벽": "벽",
    "소파": "벽",

    "복도": "복도",
}

RSSI_VMIN = -75
RSSI_VMAX = -35


# =========================
# 1. 한글 폰트 설정
# =========================

def setup_korean_font():
    system = platform.system()

    if system == "Darwin":  # macOS
        candidates = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/Library/Fonts/NanumGothic.ttf",
            "/Library/Fonts/NotoSansKR-Regular.otf",
        ]
    elif system == "Windows":
        candidates = [
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/malgunbd.ttf",
            "C:/Windows/Fonts/NanumGothic.ttf",
        ]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        ]

    selected = None
    for path in candidates:
        if Path(path).exists():
            selected = path
            break

    if selected is None:
        print("경고: 한글 폰트를 찾지 못했습니다.")
        print("macOS: Apple SD Gothic Neo / Windows: Malgun Gothic / Linux: NanumGothic 권장")
        matplotlib.rcParams["axes.unicode_minus"] = False
        return None

    fm.fontManager.addfont(selected)
    font_name = fm.FontProperties(fname=selected).get_name()

    matplotlib.rcParams["font.family"] = font_name
    matplotlib.rcParams["axes.unicode_minus"] = False

    print(f"한글 폰트 적용 완료: {font_name}")
    return font_name


# =========================
# 2. ZIP / TXT 파싱
# =========================

def fix_zip_name(name: str) -> str:
    try:
        fixed = name.encode("cp437").decode("utf-8")
    except Exception:
        fixed = name
    return unicodedata.normalize("NFC", fixed)


def clean_point_type(raw: str) -> str:
    raw = unicodedata.normalize("NFC", str(raw))
    raw = raw.replace(".txt", "").strip()
    raw_no_space = raw.replace(" ", "")

    if raw in POINT_ALIASES:
        return POINT_ALIASES[raw]
    if raw_no_space in POINT_ALIASES:
        return POINT_ALIASES[raw_no_space]

    return raw


def band_from_channel(channel):
    try:
        ch = int(channel)
    except Exception:
        return np.nan

    if 1 <= ch <= 14:
        return "2.4GHz"
    return "5GHz"


def observed_room_from_ssid(ssid: str):
    ssid = str(ssid).strip().replace('"', "")
    m = re.search(r"(\d{3})$", ssid)
    if not m:
        return np.nan
    return int(m.group(1))


def parse_location_from_filename(fixed_name: str):
    base = os.path.basename(fixed_name)
    base = unicodedata.normalize("NFC", base)

    m = re.match(r"(\d{3})_(.+)\.txt$", base)
    if not m:
        return None, None

    target_room = int(m.group(1))
    point_type = clean_point_type(m.group(2))

    return target_room, point_type


def parse_txt_bytes(data: bytes):
    text = data.decode("utf-8-sig", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not text:
        return pd.DataFrame()

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    if len(rows) < 2:
        return pd.DataFrame()

    data_rows = rows[1:]
    normalized_rows = []

    for row in data_rows:
        if len(row) < 3:
            continue

        row = [x.strip().replace('"', "") for x in row]
        normalized_rows.append(row)

    if not normalized_rows:
        return pd.DataFrame()

    df = pd.DataFrame(normalized_rows)

    if df.shape[1] >= 5:
        df = df.iloc[:, :5]
        df.columns = ["ssid", "bssid", "rssi", "channel", "time"]
    elif df.shape[1] == 4:
        df.columns = ["ssid", "bssid", "rssi", "channel"]
        df["time"] = np.nan
    elif df.shape[1] == 3:
        df.columns = ["ssid", "bssid", "rssi"]
        df["channel"] = np.nan
        df["time"] = np.nan
    else:
        return pd.DataFrame()

    df["ssid"] = df["ssid"].astype(str).str.strip()
    df["bssid"] = df["bssid"].astype(str).str.strip()
    df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")
    df["channel"] = pd.to_numeric(df["channel"], errors="coerce")
    df["band"] = df["channel"].apply(band_from_channel)
    df["observed_room"] = df["ssid"].apply(observed_room_from_ssid)

    df = df.dropna(subset=["rssi", "observed_room"])
    df["observed_room"] = df["observed_room"].astype(int)

    return df


def load_zip_to_long_df(zip_path):
    all_rows = []

    with zipfile.ZipFile(zip_path, "r") as z:
        for raw_name in z.namelist():
            if raw_name.startswith("__MACOSX"):
                continue
            if not raw_name.lower().endswith(".txt"):
                continue

            fixed_name = fix_zip_name(raw_name)
            target_room, point_type = parse_location_from_filename(fixed_name)

            if target_room is None:
                continue

            data = z.read(raw_name)
            df = parse_txt_bytes(data)

            if df.empty:
                continue

            df["file_name"] = fixed_name
            df["target_room"] = target_room
            df["point_type"] = point_type
            df["point_id"] = df["target_room"].astype(str) + "_" + df["point_type"].astype(str)

            all_rows.append(df)

    if not all_rows:
        raise ValueError("TXT 데이터를 읽지 못함. ZIP 파일명 또는 TXT 내부 형식을 확인해.")

    long_df = pd.concat(all_rows, ignore_index=True)

    return long_df


# =========================
# 3. 측정값 요약
# =========================

def aggregate_data(long_df):
    agg = (
        long_df
        .groupby(
            [
                "target_room",
                "point_type",
                "point_id",
                "observed_room",
                "ssid",
                "bssid",
                "band",
                "channel",
            ],
            dropna=False
        )
        .agg(
            rssi_mean=("rssi", "mean"),
            rssi_median=("rssi", "median"),
            rssi_std=("rssi", "std"),
            n_samples=("rssi", "count"),
        )
        .reset_index()
    )

    agg["is_self_ap"] = agg["target_room"] == agg["observed_room"]

    return agg


def create_summary(agg):
    points = (
        agg[["target_room", "point_type", "point_id"]]
        .drop_duplicates()
        .sort_values(["target_room", "point_type"])
        .reset_index(drop=True)
    )

    summary = points.copy()

    for band in ["2.4GHz", "5GHz"]:
        band_df = agg[agg["band"] == band].copy()

        self_df = band_df[band_df["is_self_ap"]].copy()
        self_df = (
            self_df
            .sort_values(["target_room", "point_type", "rssi_mean"], ascending=[True, True, False])
            .drop_duplicates(["target_room", "point_type"])
        )

        self_cols = self_df[[
            "target_room",
            "point_type",
            "observed_room",
            "ssid",
            "bssid",
            "channel",
            "rssi_mean",
            "rssi_median",
            "n_samples",
        ]].rename(columns={
            "observed_room": f"self_observed_room_{band}",
            "ssid": f"self_ssid_{band}",
            "bssid": f"self_bssid_{band}",
            "channel": f"self_channel_{band}",
            "rssi_mean": f"self_rssi_mean_{band}",
            "rssi_median": f"self_rssi_median_{band}",
            "n_samples": f"self_n_samples_{band}",
        })

        summary = summary.merge(self_cols, on=["target_room", "point_type"], how="left")

        strongest_df = (
            band_df
            .sort_values(["target_room", "point_type", "rssi_mean"], ascending=[True, True, False])
            .drop_duplicates(["target_room", "point_type"])
        )

        strongest_cols = strongest_df[[
            "target_room",
            "point_type",
            "observed_room",
            "ssid",
            "bssid",
            "channel",
            "rssi_mean",
            "rssi_median",
            "n_samples",
        ]].rename(columns={
            "observed_room": f"strongest_observed_room_{band}",
            "ssid": f"strongest_ssid_{band}",
            "bssid": f"strongest_bssid_{band}",
            "channel": f"strongest_channel_{band}",
            "rssi_mean": f"strongest_rssi_mean_{band}",
            "rssi_median": f"strongest_rssi_median_{band}",
            "n_samples": f"strongest_n_samples_{band}",
        })

        summary = summary.merge(strongest_cols, on=["target_room", "point_type"], how="left")

        summary[f"rssi_gap_strongest_minus_self_{band}"] = (
            summary[f"strongest_rssi_mean_{band}"] - summary[f"self_rssi_mean_{band}"]
        )

        for th in [-70, -75, -80]:
            count_df = (
                band_df[band_df["rssi_mean"] >= th]
                .groupby(["target_room", "point_type"])
                .size()
                .reset_index(name=f"ap_count_ge_{th}_{band}")
            )

            summary = summary.merge(count_df, on=["target_room", "point_type"], how="left")
            summary[f"ap_count_ge_{th}_{band}"] = summary[f"ap_count_ge_{th}_{band}"].fillna(0).astype(int)

    return summary


# =========================
# 4. Entropy Weight Method 품질 점수
# =========================

def minmax_benefit(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.max() == s.min():
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def minmax_cost(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.max() == s.min():
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s.max() - s) / (s.max() - s.min())


def entropy_weights(norm_df):
    X = norm_df.copy().astype(float)
    X = X.fillna(X.mean())

    eps = 1e-12
    col_sums = X.sum(axis=0).replace(0, eps)
    P = X / col_sums

    m = len(X)
    k = 1 / np.log(m)

    E = -k * (P * np.log(P + eps)).sum(axis=0)
    D = 1 - E

    if D.sum() == 0:
        W = pd.Series(np.ones(len(D)) / len(D), index=D.index)
    else:
        W = D / D.sum()

    return W, E


def add_quality_score(summary, band="5GHz"):
    df = summary.copy()

    rssi_col = f"self_rssi_mean_{band}"
    count_col = f"ap_count_ge_-75_{band}"
    gap_col = f"rssi_gap_strongest_minus_self_{band}"

    for col in [rssi_col, count_col, gap_col]:
        if col not in df.columns:
            df[col] = np.nan

    features = pd.DataFrame(index=df.index)
    features["RSSI_score"] = minmax_benefit(df[rssi_col])
    features["low_overlap_score"] = minmax_cost(df[count_col])
    features["low_gap_score"] = minmax_cost(df[gap_col].fillna(0))

    weights, entropy = entropy_weights(features)

    df[f"quality_score_{band}"] = (features * weights).sum(axis=1) * 100

    weight_table = pd.DataFrame({
        "feature": weights.index,
        "entropy": entropy.values,
        "weight": weights.values,
    })

    return df, weight_table


# =========================
# 5. 좌표 찍기
# =========================

def sort_summary_for_click(summary):
    point_rank = {p: i for i, p in enumerate(POINT_ORDER)}
    df = summary.copy()
    df["point_rank"] = df["point_type"].map(point_rank).fillna(99)
    df = df.sort_values(["target_room", "point_rank", "point_type"]).reset_index(drop=True)
    return df


def collect_coords_interactively(summary, floorplan_path, coords_path):
    img = plt.imread(floorplan_path)
    click_df = sort_summary_for_click(summary)

    coords = []

    print("\n좌표 찍기 시작")
    print("지도 창에서 제목에 나오는 측정점을 순서대로 클릭해.")
    print("예: 401_공유기, 401_문앞, 401_벽, 401_복도 ...")
    print("잘못 찍으면 나중에 coords.csv에서 x,y를 직접 수정하면 됨.\n")

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)
    ax.axis("off")

    for idx, row in click_df.iterrows():
        room = row["target_room"]
        point_type = row["point_type"]
        label = f"{room}_{point_type}"

        ax.set_title(f"클릭할 지점: {label}    ({idx + 1}/{len(click_df)})", fontsize=16)
        fig.canvas.draw()

        pts = plt.ginput(1, timeout=-1)

        if len(pts) == 0:
            x, y = np.nan, np.nan
            print(f"SKIP: {label}")
        else:
            x, y = pts[0]
            ax.scatter([x], [y], s=60, c="cyan", edgecolors="black")
            ax.text(
                x,
                y - 10,
                label,
                fontsize=7,
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75),
            )
            print(f"{label}: x={x:.1f}, y={y:.1f}")

        coords.append({
            "target_room": room,
            "point_type": point_type,
            "point_id": row["point_id"],
            "x": x,
            "y": y,
        })

    plt.close(fig)

    coords_df = pd.DataFrame(coords)
    coords_df.to_csv(coords_path, index=False, encoding="utf-8-sig")

    print(f"\n좌표 저장 완료: {coords_path}")


def ensure_coords(summary, floorplan_path):
    coords_path = OUT_DIR / "coords.csv"
    template_path = OUT_DIR / "coords_template.csv"

    if coords_path.exists():
        print(f"기존 좌표 파일 사용: {coords_path}")
        return coords_path

    template = sort_summary_for_click(summary)[["target_room", "point_type", "point_id"]].copy()
    template["x"] = np.nan
    template["y"] = np.nan
    template.to_csv(template_path, index=False, encoding="utf-8-sig")

    collect_coords_interactively(summary, floorplan_path, coords_path)

    return coords_path


# =========================
# 6. 히트맵 생성
# =========================

def make_point_overlay(summary, coords_path, floorplan_path, value_col, title, out_name,
                       vmin=None, vmax=None, cmap="RdYlGn"):
    img = plt.imread(floorplan_path)

    coords = pd.read_csv(coords_path)
    coords["point_type"] = coords["point_type"].apply(clean_point_type)

    df = summary.merge(coords, on=["target_room", "point_type", "point_id"], how="inner")
    df = df.dropna(subset=["x", "y", value_col])

    if df.empty:
        print(f"{value_col}: 표시할 데이터 없음")
        return

    if vmin is None:
        vmin = float(df[value_col].min())
    if vmax is None:
        vmax = float(df[value_col].max())

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)
    ax.axis("off")

    sc = ax.scatter(
        df["x"],
        df["y"],
        c=df[value_col],
        s=190,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.8,
        alpha=0.95,
    )

    for _, r in df.iterrows():
        ax.text(
            r["x"],
            r["y"] - 14,
            f'{int(r["target_room"])}_{r["point_type"]}\n{r[value_col]:.1f}',
            ha="center",
            va="bottom",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.72),
        )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.015)
    cbar.set_label(value_col)

    ax.set_title(title, fontsize=15)

    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=220)
    plt.close(fig)

    print(f"저장 완료: {OUT_DIR / out_name}")


def inverse_distance_grid(x, y, values, width, height, power=2.0, radius=95):
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    numerator = np.zeros((height, width), dtype=float)
    denominator = np.zeros((height, width), dtype=float)

    for xi, yi, vi in zip(x, y, values):
        dist = np.sqrt((grid_x - xi) ** 2 + (grid_y - yi) ** 2)
        dist = np.maximum(dist, 1.0)

        weight = 1 / (dist ** power)
        weight[dist > radius] = 0

        numerator += weight * vi
        denominator += weight

    grid = numerator / np.maximum(denominator, 1e-12)

    if np.nanmax(denominator) == 0:
        alpha = np.zeros_like(denominator)
    else:
        alpha = np.clip(denominator / np.nanmax(denominator), 0, 1)
        alpha = alpha ** 0.35
        alpha[denominator <= 0] = 0

    return grid, alpha


def make_smooth_overlay(summary, coords_path, floorplan_path, value_col, title, out_name,
                        vmin=None, vmax=None, radius=95, cmap="RdYlGn"):
    img = plt.imread(floorplan_path)
    height, width = img.shape[:2]

    coords = pd.read_csv(coords_path)
    coords["point_type"] = coords["point_type"].apply(clean_point_type)

    df = summary.merge(coords, on=["target_room", "point_type", "point_id"], how="inner")
    df = df.dropna(subset=["x", "y", value_col])

    if df.empty:
        print(f"{value_col}: 표시할 데이터 없음")
        return

    if vmin is None:
        vmin = float(df[value_col].min())
    if vmax is None:
        vmax = float(df[value_col].max())

    grid, alpha = inverse_distance_grid(
        x=df["x"].values,
        y=df["y"].values,
        values=df[value_col].values,
        width=width,
        height=height,
        power=2.0,
        radius=radius,
    )

    alpha = np.clip(alpha * 0.58, 0, 0.58)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    ax.axis("off")

    sc = ax.scatter(
        df["x"],
        df["y"],
        c=df[value_col],
        s=70,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.5,
    )

    for _, r in df.iterrows():
        ax.text(
            r["x"],
            r["y"] - 9,
            f'{r[value_col]:.0f}',
            ha="center",
            va="bottom",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=0.6),
        )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.015)
    cbar.set_label(value_col)

    ax.set_title(title, fontsize=15)

    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=220)
    plt.close(fig)

    print(f"저장 완료: {OUT_DIR / out_name}")


# =========================
# 7. 메인 실행
# =========================

def main():
    setup_korean_font()

    zip_path = Path(ZIP_PATH)
    floorplan_path = Path(FLOORPLAN_IMAGE)

    if not zip_path.exists():
        zips = list(Path(".").glob("*.zip"))
        if not zips:
            raise FileNotFoundError("ZIP 파일이 없음. 인수_data.zip을 같은 폴더에 넣어.")
        zip_path = zips[0]

    if not floorplan_path.exists():
        images = list(Path(".").glob("*.png")) + list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.jpeg"))
        if not images:
            raise FileNotFoundError("지도 이미지가 없음. FLOORPLAN_IMAGE 파일명을 확인해.")
        floorplan_path = images[0]

    print(f"ZIP 사용: {zip_path}")
    print(f"지도 사용: {floorplan_path}")

    long_df = load_zip_to_long_df(zip_path)
    long_df.to_csv(OUT_DIR / "wifi_long_raw.csv", index=False, encoding="utf-8-sig")

    agg = aggregate_data(long_df)
    agg.to_csv(OUT_DIR / "wifi_ap_aggregated.csv", index=False, encoding="utf-8-sig")

    summary = create_summary(agg)
    summary, weight_table = add_quality_score(summary, band="5GHz")

    summary.to_csv(OUT_DIR / "wifi_point_summary.csv", index=False, encoding="utf-8-sig")
    weight_table.to_csv(OUT_DIR / "entropy_weights_5GHz.csv", index=False, encoding="utf-8-sig")

    coords_path = ensure_coords(summary, floorplan_path)

    make_point_overlay(
        summary=summary,
        coords_path=coords_path,
        floorplan_path=floorplan_path,
        value_col="self_rssi_mean_5GHz",
        title="5GHz 자기 AP 기준 RSSI 점 히트맵",
        out_name="floorplan_point_self_rssi_5GHz.png",
        vmin=RSSI_VMIN,
        vmax=RSSI_VMAX,
        cmap="RdYlGn",
    )

    make_smooth_overlay(
        summary=summary,
        coords_path=coords_path,
        floorplan_path=floorplan_path,
        value_col="self_rssi_mean_5GHz",
        title="5GHz 자기 AP 기준 RSSI 연속 히트맵",
        out_name="floorplan_smooth_self_rssi_5GHz.png",
        vmin=RSSI_VMIN,
        vmax=RSSI_VMAX,
        radius=95,
        cmap="RdYlGn",
    )

    make_point_overlay(
        summary=summary,
        coords_path=coords_path,
        floorplan_path=floorplan_path,
        value_col="strongest_rssi_mean_5GHz",
        title="5GHz 최강 AP 기준 RSSI 점 히트맵",
        out_name="floorplan_point_strongest_rssi_5GHz.png",
        vmin=RSSI_VMIN,
        vmax=RSSI_VMAX,
        cmap="RdYlGn",
    )

    make_smooth_overlay(
        summary=summary,
        coords_path=coords_path,
        floorplan_path=floorplan_path,
        value_col="strongest_rssi_mean_5GHz",
        title="5GHz 최강 AP 기준 RSSI 연속 히트맵",
        out_name="floorplan_smooth_strongest_rssi_5GHz.png",
        vmin=RSSI_VMIN,
        vmax=RSSI_VMAX,
        radius=95,
        cmap="RdYlGn",
    )

    make_point_overlay(
        summary=summary,
        coords_path=coords_path,
        floorplan_path=floorplan_path,
        value_col="ap_count_ge_-75_5GHz",
        title="5GHz -75dBm 이상 감지 AP 개수",
        out_name="floorplan_point_ap_count_ge_-75_5GHz.png",
        vmin=0,
        vmax=None,
        cmap="viridis",
    )

    make_point_overlay(
        summary=summary,
        coords_path=coords_path,
        floorplan_path=floorplan_path,
        value_col="rssi_gap_strongest_minus_self_5GHz",
        title="5GHz 최강 AP - 자기 AP RSSI 차이",
        out_name="floorplan_point_rssi_gap_5GHz.png",
        vmin=0,
        vmax=None,
        cmap="magma",
    )

    make_point_overlay(
        summary=summary,
        coords_path=coords_path,
        floorplan_path=floorplan_path,
        value_col="quality_score_5GHz",
        title="5GHz Entropy 기반 Wi-Fi 종합 품질 점수",
        out_name="floorplan_point_quality_score_5GHz.png",
        vmin=0,
        vmax=100,
        cmap="RdYlGn",
    )

    make_smooth_overlay(
        summary=summary,
        coords_path=coords_path,
        floorplan_path=floorplan_path,
        value_col="quality_score_5GHz",
        title="5GHz Entropy 기반 Wi-Fi 종합 품질 점수 연속 히트맵",
        out_name="floorplan_smooth_quality_score_5GHz.png",
        vmin=0,
        vmax=100,
        radius=95,
        cmap="RdYlGn",
    )

    print("\n완료")
    print(f"결과 폴더: {OUT_DIR.resolve()}")
    print("생성 파일:")
    print("- wifi_long_raw.csv")
    print("- wifi_ap_aggregated.csv")
    print("- wifi_point_summary.csv")
    print("- entropy_weights_5GHz.csv")
    print("- coords.csv")
    print("- floorplan_point_self_rssi_5GHz.png")
    print("- floorplan_smooth_self_rssi_5GHz.png")
    print("- floorplan_point_strongest_rssi_5GHz.png")
    print("- floorplan_smooth_strongest_rssi_5GHz.png")
    print("- floorplan_point_ap_count_ge_-75_5GHz.png")
    print("- floorplan_point_rssi_gap_5GHz.png")
    print("- floorplan_point_quality_score_5GHz.png")
    print("- floorplan_smooth_quality_score_5GHz.png")


if __name__ == "__main__":
    main()