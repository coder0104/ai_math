# wifi_ap_overlap_kmeans.py
# 목적:
# AP 바로 아래(공유기 지점)에서 측정한 주변 AP RSSI만 사용하여
# 1) AP 간 RSSI 행렬 생성
# 2) AP 중복 커버리지 개수 계산
# 3) K-means로 AP 위치 유형 분류
# 4) 중복 과밀형 / 균형형 / 희소 커버리지형 원인 분석
# 5) 지도 위에 결과 시각화
#
# 필요 패키지:
# pip install pandas numpy matplotlib

import os
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

INPUT_DIR = Path("wifi_floorplan_outputs")
OUT_DIR = Path("wifi_ap_overlap_outputs")
OUT_DIR.mkdir(exist_ok=True)

LONG_RAW_PATH = INPUT_DIR / "wifi_long_raw.csv"
COORDS_PATH = INPUT_DIR / "coords.csv"

# 네 지도 이미지 파일명
FLOORPLAN_IMAGE = "KakaoTalk_Photo_2026-06-15-20-26-23.jpeg"

BAND = "5GHz"

# 중복 커버리지 판단 기준
OVERLAP_THRESHOLD = -75
STRONG_OVERLAP_THRESHOLD = -65

# 너무 약하게 잡힌 AP까지 평균에 넣으면 왜곡될 수 있어서 설정
NEIGHBOR_RSSI_MIN = -85

# K-means 설정
K = 3
RANDOM_SEED = 42
MAX_ITER = 200


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
    else:
        candidates = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]

    selected = None
    for p in candidates:
        if Path(p).exists():
            selected = p
            break

    if selected is not None:
        fm.fontManager.addfont(selected)
        font_name = fm.FontProperties(fname=selected).get_name()
        matplotlib.rcParams["font.family"] = font_name
        print(f"한글 폰트 적용: {font_name}")
    else:
        print("경고: 한글 폰트를 찾지 못했습니다.")

    matplotlib.rcParams["axes.unicode_minus"] = False


# =========================
# 2. 데이터 로드 및 정리
# =========================

def normalize_point_type(x):
    x = str(x).strip()

    aliases = {
        "공유기": "공유기",
        "AP": "공유기",
        "ap": "공유기",
        "책상": "공유기",
        "문앞": "문앞",
        "문쪽": "문앞",
        "문": "문앞",
        "소파쪽 문": "문앞",
        "소파쪽문": "문앞",
        "벽": "벽",
        "소파": "벽",
        "복도": "복도",
    }

    return aliases.get(x, x)


def load_data():
    if not LONG_RAW_PATH.exists():
        raise FileNotFoundError(
            f"{LONG_RAW_PATH} 파일이 없습니다. 먼저 wifi_floorplan_heatmap 코드를 실행하세요."
        )

    df = pd.read_csv(LONG_RAW_PATH)
    df["point_type"] = df["point_type"].apply(normalize_point_type)

    needed = [
        "target_room",
        "point_type",
        "observed_room",
        "rssi",
        "channel",
        "band",
        "ssid",
        "bssid",
    ]

    for col in needed:
        if col not in df.columns:
            raise ValueError(f"필수 열이 없습니다: {col}")

    df = df[df["band"] == BAND].copy()
    df = df[df["point_type"] == "공유기"].copy()

    df["target_room"] = pd.to_numeric(df["target_room"], errors="coerce")
    df["observed_room"] = pd.to_numeric(df["observed_room"], errors="coerce")
    df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")
    df["channel"] = pd.to_numeric(df["channel"], errors="coerce")

    df = df.dropna(subset=["target_room", "observed_room", "rssi"])
    df["target_room"] = df["target_room"].astype(int)
    df["observed_room"] = df["observed_room"].astype(int)

    return df


def load_coords():
    if not COORDS_PATH.exists():
        print("coords.csv가 없어 지도 오버레이는 생략됩니다.")
        return None

    coords = pd.read_csv(COORDS_PATH)
    coords["point_type"] = coords["point_type"].apply(normalize_point_type)
    coords = coords[coords["point_type"] == "공유기"].copy()

    coords["target_room"] = pd.to_numeric(coords["target_room"], errors="coerce")
    coords = coords.dropna(subset=["target_room", "x", "y"])
    coords["target_room"] = coords["target_room"].astype(int)

    return coords[["target_room", "x", "y"]]


# =========================
# 3. AP 간 RSSI 행렬 생성
# =========================

def build_ap_rssi_table(df):
    # 같은 target_room에서 같은 observed_room이 여러 번 잡힌 경우 평균 RSSI 사용
    ap_table = (
        df.groupby(["target_room", "observed_room"], as_index=False)
        .agg(
            rssi_mean=("rssi", "mean"),
            rssi_max=("rssi", "max"),
            rssi_min=("rssi", "min"),
            n_samples=("rssi", "count"),
            channel_mode=("channel", lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan),
        )
    )

    ap_table.to_csv(OUT_DIR / "ap_under_raw_aggregated.csv", index=False, encoding="utf-8-sig")
    return ap_table


def make_rssi_matrix(ap_table):
    rooms = sorted(ap_table["target_room"].unique())
    observed = sorted(ap_table["observed_room"].unique())

    matrix = pd.DataFrame(index=rooms, columns=observed, dtype=float)

    for _, row in ap_table.iterrows():
        matrix.loc[row["target_room"], row["observed_room"]] = row["rssi_mean"]

    matrix.to_csv(OUT_DIR / f"ap_to_ap_rssi_matrix_{BAND}.csv", encoding="utf-8-sig")
    return matrix


def make_overlap_matrix(rssi_matrix, threshold=OVERLAP_THRESHOLD):
    overlap = (rssi_matrix >= threshold).astype(int)

    # 자기 자신 AP는 중복 커버리지 계산에서 제외
    for room in overlap.index:
        if room in overlap.columns:
            overlap.loc[room, room] = 0

    overlap.to_csv(
        OUT_DIR / f"ap_overlap_binary_matrix_ge_{threshold}_{BAND}.csv",
        encoding="utf-8-sig"
    )
    return overlap


# =========================
# 4. AP별 원인 분석 feature 생성
# =========================

def get_self_channel(ap_table):
    self_rows = ap_table[ap_table["target_room"] == ap_table["observed_room"]].copy()
    channel_map = dict(zip(self_rows["target_room"], self_rows["channel_mode"]))
    return channel_map


def build_features(ap_table, rssi_matrix):
    rooms = list(rssi_matrix.index)
    channel_map = get_self_channel(ap_table)

    rows = []

    for room in rooms:
        row_rssi = rssi_matrix.loc[room]

        self_rssi = row_rssi.get(room, np.nan)

        neighbor_rssi = row_rssi.drop(labels=[room], errors="ignore").dropna()
        meaningful_neighbors = neighbor_rssi[neighbor_rssi >= NEIGHBOR_RSSI_MIN]

        overlap_count_m75 = int((neighbor_rssi >= OVERLAP_THRESHOLD).sum())
        strong_overlap_count_m65 = int((neighbor_rssi >= STRONG_OVERLAP_THRESHOLD).sum())

        mean_neighbor_rssi = float(meaningful_neighbors.mean()) if len(meaningful_neighbors) else np.nan
        max_neighbor_rssi = float(meaningful_neighbors.max()) if len(meaningful_neighbors) else np.nan

        # 같은 채널 중복 개수
        self_channel = channel_map.get(room, np.nan)
        same_channel_overlap = 0

        for observed_room in neighbor_rssi.index:
            rssi_value = neighbor_rssi.loc[observed_room]
            if pd.isna(rssi_value) or rssi_value < OVERLAP_THRESHOLD:
                continue

            temp = ap_table[
                (ap_table["target_room"] == room)
                & (ap_table["observed_room"] == observed_room)
            ]

            if len(temp) == 0:
                continue

            observed_channel = temp["channel_mode"].iloc[0]

            if not pd.isna(self_channel) and not pd.isna(observed_channel):
                if int(self_channel) == int(observed_channel):
                    same_channel_overlap += 1

        # 자기 AP가 너무 약한지
        self_weak_flag = int(self_rssi <= -55) if not pd.isna(self_rssi) else 0

        # 주변 AP가 거의 안 잡히는지
        sparse_flag = int(overlap_count_m75 <= 1)

        # 주변 AP가 과도하게 겹치는지
        dense_flag = int(overlap_count_m75 >= 3)

        rows.append({
            "ap_room": room,
            "self_rssi": self_rssi,
            "overlap_count_ge_-75": overlap_count_m75,
            "strong_overlap_count_ge_-65": strong_overlap_count_m65,
            "same_channel_overlap_ge_-75": same_channel_overlap,
            "mean_neighbor_rssi": mean_neighbor_rssi,
            "max_neighbor_rssi": max_neighbor_rssi,
            "self_channel": self_channel,
            "self_weak_flag": self_weak_flag,
            "sparse_flag": sparse_flag,
            "dense_flag": dense_flag,
        })

    features = pd.DataFrame(rows)

    features.to_csv(OUT_DIR / f"ap_overlap_features_{BAND}.csv", index=False, encoding="utf-8-sig")
    return features


# =========================
# 5. K-means 직접 구현
# =========================

def standardize_features(X):
    X = X.astype(float)
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0

    X_filled = np.where(np.isnan(X), mean, X)
    Z = (X_filled - mean) / std

    return Z, mean, std


def init_centroids_farthest(Z, k):
    np.random.seed(RANDOM_SEED)

    n = len(Z)
    first = np.random.randint(0, n)
    centroids = [Z[first]]

    while len(centroids) < k:
        dists = np.array([
            min(np.linalg.norm(z - c) for c in centroids)
            for z in Z
        ])
        next_idx = int(np.argmax(dists))
        centroids.append(Z[next_idx])

    return np.array(centroids)


def run_kmeans(Z, k=3, max_iter=200):
    n = len(Z)
    k = min(k, n)

    centroids = init_centroids_farthest(Z, k)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        old_labels = labels.copy()

        dist_matrix = np.array([
            [np.linalg.norm(z - c) for c in centroids]
            for z in Z
        ])

        labels = np.argmin(dist_matrix, axis=1)

        for c in range(k):
            members = Z[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)

        if np.array_equal(labels, old_labels):
            break

    inertia = 0.0
    for i, z in enumerate(Z):
        inertia += np.linalg.norm(z - centroids[labels[i]]) ** 2

    return labels, centroids, inertia


def assign_cluster_names(features):
    cluster_summary = (
        features.groupby("cluster")
        .agg(
            mean_overlap=("overlap_count_ge_-75", "mean"),
            mean_strong_overlap=("strong_overlap_count_ge_-65", "mean"),
            mean_same_channel=("same_channel_overlap_ge_-75", "mean"),
            mean_self_rssi=("self_rssi", "mean"),
            count=("ap_room", "count"),
        )
        .reset_index()
    )

    # 중복 개수가 가장 큰 군집 = 중복 과밀형
    dense_cluster = cluster_summary.sort_values(
        ["mean_overlap", "mean_same_channel", "mean_strong_overlap"],
        ascending=False
    )["cluster"].iloc[0]

    # 중복 개수가 가장 작은 군집 = 희소 커버리지형
    sparse_cluster = cluster_summary.sort_values(
        ["mean_overlap", "mean_strong_overlap"],
        ascending=True
    )["cluster"].iloc[0]

    name_map = {}
    for c in cluster_summary["cluster"]:
        if c == dense_cluster:
            name_map[c] = "중복 과밀형"
        elif c == sparse_cluster:
            name_map[c] = "희소 커버리지형"
        else:
            name_map[c] = "균형형"

    features["cluster_type"] = features["cluster"].map(name_map)

    cluster_summary["cluster_type"] = cluster_summary["cluster"].map(name_map)

    return features, cluster_summary


def kmeans_analysis(features):
    feature_cols = [
        "self_rssi",
        "overlap_count_ge_-75",
        "strong_overlap_count_ge_-65",
        "same_channel_overlap_ge_-75",
        "mean_neighbor_rssi",
    ]

    X = features[feature_cols].values
    Z, mean, std = standardize_features(X)

    labels, centroids, inertia = run_kmeans(Z, k=K, max_iter=MAX_ITER)

    result = features.copy()
    result["cluster"] = labels

    result, cluster_summary = assign_cluster_names(result)

    result.to_csv(OUT_DIR / f"ap_kmeans_result_{BAND}.csv", index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(OUT_DIR / f"ap_kmeans_cluster_summary_{BAND}.csv", index=False, encoding="utf-8-sig")

    # K 선택 참고용 elbow
    elbow_rows = []
    max_k = min(6, len(features))

    for k in range(1, max_k + 1):
        labels_k, centroids_k, inertia_k = run_kmeans(Z, k=k, max_iter=MAX_ITER)
        elbow_rows.append({"k": k, "inertia": inertia_k})

    elbow_df = pd.DataFrame(elbow_rows)
    elbow_df.to_csv(OUT_DIR / "kmeans_elbow.csv", index=False, encoding="utf-8-sig")

    return result, cluster_summary, elbow_df


# =========================
# 6. 시각화
# =========================

def plot_rssi_matrix(matrix):
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix.values.astype(float), vmin=-85, vmax=-30, cmap="RdYlGn")

    ax.set_title(f"AP 밑 측정값 기반 AP 간 RSSI 행렬 ({BAND})")
    ax.set_xlabel("감지된 AP 번호")
    ax.set_ylabel("측정 위치 AP 번호")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.isna(val):
                text = "X"
            else:
                text = f"{val:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RSSI(dBm)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"ap_rssi_matrix_heatmap_{BAND}.png", dpi=220)
    plt.close(fig)


def plot_overlap_matrix(overlap):
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(overlap.values.astype(float), vmin=0, vmax=1, cmap="Greys")

    ax.set_title(f"AP 중복 커버리지 행렬: RSSI ≥ {OVERLAP_THRESHOLD}dBm ({BAND})")
    ax.set_xlabel("감지된 주변 AP 번호")
    ax.set_ylabel("측정 위치 AP 번호")

    ax.set_xticks(range(len(overlap.columns)))
    ax.set_xticklabels(overlap.columns, rotation=90)
    ax.set_yticks(range(len(overlap.index)))
    ax.set_yticklabels(overlap.index)

    for i in range(overlap.shape[0]):
        for j in range(overlap.shape[1]):
            val = overlap.iloc[i, j]
            ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("중복 여부")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"ap_overlap_binary_heatmap_{BAND}.png", dpi=220)
    plt.close(fig)


def plot_kmeans_elbow(elbow_df):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
    ax.set_title("K-means K값 선택 참고 Elbow Graph")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "kmeans_elbow_graph.png", dpi=220)
    plt.close(fig)


def plot_feature_scatter(result):
    fig, ax = plt.subplots(figsize=(8, 6))

    cluster_types = result["cluster_type"].unique()

    for ct in cluster_types:
        part = result[result["cluster_type"] == ct]
        ax.scatter(
            part["overlap_count_ge_-75"],
            part["self_rssi"],
            s=180,
            edgecolors="black",
            label=ct
        )

        for _, r in part.iterrows():
            ax.text(
                r["overlap_count_ge_-75"],
                r["self_rssi"] + 0.7,
                str(int(r["ap_room"])),
                ha="center",
                fontsize=8
            )

    ax.set_title("AP 위치 유형 분류: 중복 AP 개수 vs 자기 AP RSSI")
    ax.set_xlabel(f"주변 AP 중복 개수 (RSSI ≥ {OVERLAP_THRESHOLD}dBm)")
    ax.set_ylabel("자기 AP RSSI(dBm)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ap_kmeans_feature_scatter.png", dpi=220)
    plt.close(fig)


def find_floorplan():
    p = Path(FLOORPLAN_IMAGE)
    if p.exists():
        return p

    candidates = list(Path(".").glob("*.png")) + list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.jpeg"))
    if candidates:
        return candidates[0]

    return None


def plot_floorplan_overlay(result, coords):
    if coords is None:
        return

    floorplan = find_floorplan()
    if floorplan is None:
        print("지도 이미지가 없어 오버레이 생략")
        return

    img = plt.imread(floorplan)

    df = result.merge(coords, left_on="ap_room", right_on="target_room", how="left")
    df = df.dropna(subset=["x", "y"])

    if df.empty:
        print("좌표가 없어 오버레이 생략")
        return

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)
    ax.axis("off")

    markers = {
        "중복 과밀형": "X",
        "균형형": "o",
        "희소 커버리지형": "^",
    }

    for ct in ["중복 과밀형", "균형형", "희소 커버리지형"]:
        part = df[df["cluster_type"] == ct]
        if part.empty:
            continue

        ax.scatter(
            part["x"],
            part["y"],
            s=260,
            marker=markers.get(ct, "o"),
            edgecolors="black",
            linewidths=1.0,
            label=ct
        )

        for _, r in part.iterrows():
            ax.text(
                r["x"],
                r["y"] - 18,
                f"{int(r['ap_room'])}\n{ct}\n중복:{int(r['overlap_count_ge_-75'])}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75)
            )

    ax.set_title(f"K-means 기반 AP 위치 유형 분류 지도 ({BAND})", fontsize=15)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"ap_kmeans_floorplan_overlay_{BAND}.png", dpi=220)
    plt.close(fig)


# =========================
# 7. 원인 분석 보고서 생성
# =========================

def create_report(result, cluster_summary):
    dense = result[result["cluster_type"] == "중복 과밀형"].copy()
    sparse = result[result["cluster_type"] == "희소 커버리지형"].copy()
    balanced = result[result["cluster_type"] == "균형형"].copy()

    lines = []

    lines.append("AP 밑 측정값 기반 Wi-Fi 근본 원인 분석 결과")
    lines.append("")
    lines.append(f"분석 대역: {BAND}")
    lines.append(f"중복 커버리지 기준: 주변 AP RSSI ≥ {OVERLAP_THRESHOLD}dBm")
    lines.append(f"강한 중복 기준: 주변 AP RSSI ≥ {STRONG_OVERLAP_THRESHOLD}dBm")
    lines.append("")
    lines.append("[1] 분석 방법")
    lines.append(
        "각 교실의 AP 바로 아래에서 측정한 Wi-Fi 스캔 데이터를 사용하였다. "
        "파일 내부에 기록된 주변 AP의 RSSI를 이용해 AP 간 RSSI 행렬을 구성하고, "
        f"특정 AP 위치에서 다른 AP가 {OVERLAP_THRESHOLD}dBm 이상으로 감지되면 "
        "중복 커버리지가 존재한다고 정의하였다. 이후 자기 AP RSSI, 주변 AP 중복 개수, "
        "강한 중복 개수, 같은 채널 중복 개수, 주변 AP 평균 RSSI를 feature로 구성하고 "
        "K-means 군집화를 적용하였다."
    )
    lines.append("")
    lines.append("[2] K-means 군집 요약")

    for _, r in cluster_summary.iterrows():
        lines.append(
            f"- Cluster {int(r['cluster'])} ({r['cluster_type']}): "
            f"{int(r['count'])}개 AP, "
            f"평균 중복 AP 수 {r['mean_overlap']:.2f}, "
            f"평균 강한 중복 AP 수 {r['mean_strong_overlap']:.2f}, "
            f"평균 같은 채널 중복 {r['mean_same_channel']:.2f}, "
            f"평균 자기 AP RSSI {r['mean_self_rssi']:.1f}dBm"
        )

    lines.append("")
    lines.append("[3] 유형별 원인 해석")

    if not dense.empty:
        dense_rooms = ", ".join(str(int(x)) for x in dense["ap_room"].tolist())
        lines.append(
            f"- 중복 과밀형 AP: {dense_rooms}. "
            "이 구간은 AP 신호가 부족한 구간이라기보다, 주변 AP가 동시에 많이 감지되는 구간이다. "
            "따라서 추가 공유기 설치보다 채널 중복 여부 확인, AP 출력 조정, 로밍 기준 조정이 우선이다."
        )

    if not sparse.empty:
        sparse_rooms = ", ".join(str(int(x)) for x in sparse["ap_room"].tolist())
        lines.append(
            f"- 희소 커버리지형 AP: {sparse_rooms}. "
            "이 구간은 주변 AP가 상대적으로 적게 감지되는 구간이다. "
            "자기 AP 자체가 고장나거나 순간적으로 부하가 걸릴 경우 대체 커버리지가 부족할 수 있다. "
            "복도나 경계부 보조 AP 설치 후보로 검토할 수 있다."
        )

    if not balanced.empty:
        balanced_rooms = ", ".join(str(int(x)) for x in balanced["ap_room"].tolist())
        lines.append(
            f"- 균형형 AP: {balanced_rooms}. "
            "자기 AP 신호와 주변 AP 중복 정도가 비교적 균형적인 구간이다. "
            "우선적인 장비 추가 대상은 아니다."
        )

    lines.append("")
    lines.append("[4] 최종 결론")
    lines.append(
        "본 분석 결과, Wi-Fi가 안 터지는 원인을 단순히 공유기 수 부족으로 해석하기는 어렵다. "
        "AP 바로 아래에서도 주변 AP가 함께 감지되는 정도가 구역마다 다르게 나타났으며, "
        "일부 구간은 중복 커버리지가 많아 간섭 가능성이 있고, 일부 구간은 주변 AP가 적어 "
        "보조 커버리지가 부족한 구조로 나타났다. 따라서 학교 4층의 근본 원인은 "
        "절대적 신호 부족 하나가 아니라 AP 배치 밀도 차이, 중복 커버리지, 같은 채널 간섭 가능성, "
        "공간 구조에 따른 신호 감쇄가 결합된 문제로 해석하는 것이 타당하다."
    )

    report = "\n".join(lines)

    with open(OUT_DIR / "ap_overlap_kmeans_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(report)


# =========================
# 8. 실행
# =========================

def main():
    setup_korean_font()

    df = load_data()
    coords = load_coords()

    print(f"AP 밑 측정 데이터 수: {len(df)}")

    ap_table = build_ap_rssi_table(df)
    rssi_matrix = make_rssi_matrix(ap_table)
    overlap_matrix = make_overlap_matrix(rssi_matrix, threshold=OVERLAP_THRESHOLD)

    features = build_features(ap_table, rssi_matrix)
    result, cluster_summary, elbow_df = kmeans_analysis(features)

    plot_rssi_matrix(rssi_matrix)
    plot_overlap_matrix(overlap_matrix)
    plot_kmeans_elbow(elbow_df)
    plot_feature_scatter(result)
    plot_floorplan_overlay(result, coords)

    create_report(result, cluster_summary)

    print("\n완료")
    print(f"결과 폴더: {OUT_DIR.resolve()}")
    print("생성 파일:")
    print("- ap_under_raw_aggregated.csv")
    print("- ap_to_ap_rssi_matrix_5GHz.csv")
    print("- ap_overlap_binary_matrix_ge_-75_5GHz.csv")
    print("- ap_overlap_features_5GHz.csv")
    print("- ap_kmeans_result_5GHz.csv")
    print("- ap_kmeans_cluster_summary_5GHz.csv")
    print("- ap_rssi_matrix_heatmap_5GHz.png")
    print("- ap_overlap_binary_heatmap_5GHz.png")
    print("- kmeans_elbow_graph.png")
    print("- ap_kmeans_feature_scatter.png")
    print("- ap_kmeans_floorplan_overlay_5GHz.png")
    print("- ap_overlap_kmeans_report.txt")


if __name__ == "__main__":
    main()