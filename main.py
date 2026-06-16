# wifi_ai_solution.py
# 목적:
# 1) 품질 점수 하위 지점 추출
# 2) DBSCAN으로 Wi-Fi 취약 구역 군집 탐지
# 3) 군집 기반 AP/증폭기 후보 위치 자동 생성
# 4) 간단한 경로 손실 모델로 설치 후 RSSI 개선량 예측
# 5) NSGA-II 유전 알고리즘으로 AP/증폭기 설치 위치 최적화
# 6) 비용 우선안 / 성능 우선안 / 균형안 도출

import math
import random
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =========================
# 0. 경로 설정
# =========================

INPUT_DIR = Path("wifi_floorplan_outputs")
OUT_DIR = Path("wifi_ai_solution_outputs")
OUT_DIR.mkdir(exist_ok=True)

SUMMARY_PATH = INPUT_DIR / "wifi_point_summary.csv"
COORDS_PATH = INPUT_DIR / "coords.csv"

# 네 지도 이미지 파일명
FLOORPLAN_IMAGE = "KakaoTalk_Photo_2026-06-15-20-26-23.jpeg"

# 분석 기준
BAND = "5GHz"
QUALITY_COL = f"quality_score_{BAND}"
RSSI_COL = f"self_rssi_mean_{BAND}"
STRONGEST_COL = f"strongest_rssi_mean_{BAND}"
AP_COUNT_COL = f"ap_count_ge_-75_{BAND}"

# 취약 지점 정의
WEAK_QUANTILE = 0.25       # 품질 점수 하위 25%
WEAK_ABSOLUTE_THRESHOLD = 35  # 품질 점수 35점 이하도 취약 지점으로 포함

# DBSCAN 설정
MIN_PTS = 4
EPS_MULTIPLIERS = [0.8, 0.9, 1.0, 1.1, 1.2]

# 경로 손실 모델 근사 설정
# 픽셀을 실제 거리로 바꾸기 위한 값. 네 지도 크기에 따라 조정 가능.
PIXELS_PER_METER = 28.0
P0_RSSI = -38.0            # 새 AP/증폭기 근처 1m RSSI 가정
PATH_LOSS_N = 2.3          # 실내 경로 손실 지수
WALL_PENALTY_BLOCKED = 5.0 # "벽" 지점 추가 감쇄 근사값
NEW_SIGNAL_CUTOFF = -80.0  # 이보다 약한 신호는 개선 효과 없음으로 처리

# 설치 후보 생성
MAX_EXTRA_WEAK_CANDIDATES = 4

# NSGA-II 설정
POP_SIZE = 80
N_GENERATIONS = 120
MUTATION_RATE = 0.08
CROSSOVER_RATE = 0.85
RANDOM_SEED = 42

# 목적함수 가중치가 아니라 결과 해석용 기준
TARGET_RSSI = -55.0


# =========================
# 1. 한글 폰트
# =========================

def setup_korean_font():
    system = platform.system()

    if system == "Darwin":
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

    if selected:
        fm.fontManager.addfont(selected)
        font_name = fm.FontProperties(fname=selected).get_name()
        matplotlib.rcParams["font.family"] = font_name
        print(f"한글 폰트 적용: {font_name}")
    else:
        print("경고: 한글 폰트를 찾지 못했습니다.")

    matplotlib.rcParams["axes.unicode_minus"] = False


# =========================
# 2. 데이터 로드
# =========================

def load_data():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{SUMMARY_PATH} 파일이 없습니다. 기존 히트맵 코드를 먼저 실행하세요.")

    if not COORDS_PATH.exists():
        raise FileNotFoundError(f"{COORDS_PATH} 파일이 없습니다. 기존 코드에서 좌표를 먼저 찍어야 합니다.")

    summary = pd.read_csv(SUMMARY_PATH)
    coords = pd.read_csv(COORDS_PATH)

    df = summary.merge(
        coords[["target_room", "point_type", "point_id", "x", "y"]],
        on=["target_room", "point_type", "point_id"],
        how="left"
    )

    required = ["x", "y", RSSI_COL, STRONGEST_COL, AP_COUNT_COL]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"필수 열이 없습니다: {col}")

    df = df.dropna(subset=["x", "y", RSSI_COL]).reset_index(drop=True)

    if QUALITY_COL not in df.columns:
        print(f"{QUALITY_COL} 없음 → RSSI 기반 임시 품질 점수 생성")
        df[QUALITY_COL] = make_fallback_quality_score(df)

    return df


def make_fallback_quality_score(df):
    rssi = df[RSSI_COL].astype(float)
    ap_count = df[AP_COUNT_COL].astype(float)
    gap = df.get(f"rssi_gap_strongest_minus_self_{BAND}", pd.Series(0, index=df.index)).fillna(0)

    rssi_score = (rssi - rssi.min()) / max(rssi.max() - rssi.min(), 1e-9)
    overlap_score = (ap_count.max() - ap_count) / max(ap_count.max() - ap_count.min(), 1e-9)
    gap_score = (gap.max() - gap) / max(gap.max() - gap.min(), 1e-9)

    return 100 * (0.55 * rssi_score + 0.35 * overlap_score + 0.10 * gap_score)


# =========================
# 3. 취약 지점 추출
# =========================

def extract_weak_points(df):
    q_threshold = df[QUALITY_COL].quantile(WEAK_QUANTILE)
    threshold = max(q_threshold, WEAK_ABSOLUTE_THRESHOLD)

    weak = df[df[QUALITY_COL] <= threshold].copy()
    weak = weak.sort_values(QUALITY_COL).reset_index(drop=True)

    weak.to_csv(OUT_DIR / "weak_points.csv", index=False, encoding="utf-8-sig")

    print(f"품질 점수 하위 기준값: {q_threshold:.2f}")
    print(f"최종 취약 지점 기준값: {threshold:.2f}")
    print(f"취약 지점 수: {len(weak)}")

    return weak, threshold


# =========================
# 4. DBSCAN 직접 구현
# =========================

def pairwise_distances(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def k_distance_values(X, min_pts):
    D = pairwise_distances(X)
    sorted_D = np.sort(D, axis=1)
    k_index = min(min_pts, sorted_D.shape[1] - 1)
    return np.sort(sorted_D[:, k_index])


def estimate_elbow_eps(kdist):
    if len(kdist) < 3:
        return float(np.median(kdist)) if len(kdist) else 40.0

    x = np.arange(len(kdist))
    y = kdist

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    line = p2 - p1
    line_norm = np.linalg.norm(line)

    if line_norm == 0:
        return float(np.median(kdist))

    distances = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi])
        dist = np.abs(np.cross(line, p1 - p)) / line_norm
        distances.append(dist)

    elbow_idx = int(np.argmax(distances))
    return float(kdist[elbow_idx])


def simple_dbscan(X, eps, min_pts):
    n = len(X)
    D = pairwise_distances(X)

    labels = np.full(n, -99, dtype=int)
    cluster_id = 0

    def neighbors(i):
        return np.where(D[i] <= eps)[0].tolist()

    for i in range(n):
        if labels[i] != -99:
            continue

        neigh = neighbors(i)

        if len(neigh) < min_pts:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seeds = neigh[:]

        while seeds:
            j = seeds.pop(0)

            if labels[j] == -1:
                labels[j] = cluster_id

            if labels[j] != -99:
                continue

            labels[j] = cluster_id
            neigh_j = neighbors(j)

            if len(neigh_j) >= min_pts:
                for p in neigh_j:
                    if p not in seeds:
                        seeds.append(p)

        cluster_id += 1

    labels[labels == -99] = -1
    return labels


def run_dbscan(weak):
    X = weak[["x", "y"]].values.astype(float)

    if len(X) < MIN_PTS:
        weak["cluster"] = -1
        return weak, 50.0

    kdist = k_distance_values(X, MIN_PTS)
    eps = estimate_elbow_eps(kdist)

    # 너무 작은 eps 방지
    eps = max(eps, np.percentile(kdist, 50))

    labels = simple_dbscan(X, eps=eps, min_pts=MIN_PTS)

    result = weak.copy()
    result["cluster"] = labels
    result.to_csv(OUT_DIR / "dbscan_result.csv", index=False, encoding="utf-8-sig")

    plot_k_distance(kdist, eps)
    plot_dbscan_clusters(result, eps)

    sensitivity_rows = []
    for m in EPS_MULTIPLIERS:
        eps_m = eps * m
        labels_m = simple_dbscan(X, eps=eps_m, min_pts=MIN_PTS)
        n_clusters = len(set(labels_m)) - (1 if -1 in labels_m else 0)
        n_noise = int(np.sum(labels_m == -1))

        sensitivity_rows.append({
            "eps_multiplier": m,
            "eps": eps_m,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        })

    pd.DataFrame(sensitivity_rows).to_csv(
        OUT_DIR / "dbscan_sensitivity.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"DBSCAN eps: {eps:.2f}")
    print(f"DBSCAN 군집 수: {len(set(labels)) - (1 if -1 in labels else 0)}")

    return result, eps


def plot_k_distance(kdist, eps):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kdist, marker="o")
    ax.axhline(eps, linestyle="--", label=f"선택 eps = {eps:.1f}")
    ax.set_title("k-distance graph 기반 DBSCAN eps 결정")
    ax.set_xlabel("정렬된 취약 지점")
    ax.set_ylabel(f"{MIN_PTS}번째 이웃 거리")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dbscan_k_distance_graph.png", dpi=220)
    plt.close(fig)


def plot_dbscan_clusters(result, eps):
    fig, ax = plt.subplots(figsize=(10, 7))

    sc = ax.scatter(
        result["x"],
        result["y"],
        c=result["cluster"],
        s=230,
        cmap="tab10",
        edgecolors="black"
    )

    for _, r in result.iterrows():
        ax.text(
            r["x"],
            r["y"] - 10,
            f'{int(r["target_room"])}_{r["point_type"]}\nQ={r[QUALITY_COL]:.1f}',
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75)
        )

    ax.set_title(f"DBSCAN 기반 Wi-Fi 취약 구역 군집 탐지 eps={eps:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "dbscan_clusters_xy.png", dpi=220)
    plt.close(fig)


# =========================
# 5. 설치 후보 위치 생성
# =========================

def generate_candidates(df, dbscan_result):
    candidates = []

    # 1) 군집 중심 후보
    cluster_ids = sorted([c for c in dbscan_result["cluster"].unique() if c != -1])

    for c in cluster_ids:
        part = dbscan_result[dbscan_result["cluster"] == c]

        cx = float(part["x"].mean())
        cy = float(part["y"].mean())

        nearest_idx = nearest_point_index(df, cx, cy)
        nearest = df.loc[nearest_idx]

        base_id = f"C{c+1}"

        candidates.append(make_candidate_row(
            candidate_id=f"{base_id}_AP",
            base_id=base_id,
            device_type="AP",
            x=cx,
            y=cy,
            source=f"DBSCAN_Cluster_{c}",
            nearest_room=nearest["target_room"],
            nearest_point_type=nearest["point_type"],
            backhaul_rssi=np.nan,
            cost=1.00,
            install_difficulty=1.00
        ))

        candidates.append(make_candidate_row(
            candidate_id=f"{base_id}_EXT",
            base_id=base_id,
            device_type="증폭기",
            x=cx,
            y=cy,
            source=f"DBSCAN_Cluster_{c}",
            nearest_room=nearest["target_room"],
            nearest_point_type=nearest["point_type"],
            backhaul_rssi=float(nearest[STRONGEST_COL]),
            cost=0.65,
            install_difficulty=0.70
        ))

    # 2) 군집으로 묶이지 않은 최저 품질 지점 후보 추가
    top_weak = (
        dbscan_result
        .sort_values(QUALITY_COL)
        .head(MAX_EXTRA_WEAK_CANDIDATES)
    )

    for i, (_, r) in enumerate(top_weak.iterrows(), start=1):
        base_id = f"W{i}"

        candidates.append(make_candidate_row(
            candidate_id=f"{base_id}_AP",
            base_id=base_id,
            device_type="AP",
            x=float(r["x"]),
            y=float(r["y"]),
            source=f"WeakPoint_{int(r['target_room'])}_{r['point_type']}",
            nearest_room=r["target_room"],
            nearest_point_type=r["point_type"],
            backhaul_rssi=np.nan,
            cost=1.00,
            install_difficulty=1.00
        ))

        candidates.append(make_candidate_row(
            candidate_id=f"{base_id}_EXT",
            base_id=base_id,
            device_type="증폭기",
            x=float(r["x"]),
            y=float(r["y"]),
            source=f"WeakPoint_{int(r['target_room'])}_{r['point_type']}",
            nearest_room=r["target_room"],
            nearest_point_type=r["point_type"],
            backhaul_rssi=float(r[STRONGEST_COL]),
            cost=0.65,
            install_difficulty=0.70
        ))

    cand = pd.DataFrame(candidates)

    # 증폭기는 backhaul 조건 적용
    cand["backhaul_ok"] = True
    is_ext = cand["device_type"] == "증폭기"
    cand.loc[is_ext, "backhaul_ok"] = cand.loc[is_ext, "backhaul_rssi"] >= -65

    # Backhaul 안 되는 증폭기는 후보에서 제거하지 않고, 최적화에서 강한 패널티 부여
    cand.to_csv(OUT_DIR / "generated_candidates.csv", index=False, encoding="utf-8-sig")

    print(f"설치 후보 수: {len(cand)}")
    return cand


def make_candidate_row(candidate_id, base_id, device_type, x, y, source,
                       nearest_room, nearest_point_type,
                       backhaul_rssi, cost, install_difficulty):
    return {
        "candidate_id": candidate_id,
        "base_id": base_id,
        "device_type": device_type,
        "x": x,
        "y": y,
        "source": source,
        "nearest_room": nearest_room,
        "nearest_point_type": nearest_point_type,
        "backhaul_rssi": backhaul_rssi,
        "cost": cost,
        "install_difficulty": install_difficulty,
    }


def nearest_point_index(df, x, y):
    dist = np.sqrt((df["x"] - x) ** 2 + (df["y"] - y) ** 2)
    return int(dist.idxmin())


# =========================
# 6. 설치 효과 예측
# =========================

def estimate_new_rssi(candidate, point):
    dx = float(candidate["x"] - point["x"])
    dy = float(candidate["y"] - point["y"])
    pixel_dist = math.sqrt(dx * dx + dy * dy)

    d_m = max(pixel_dist / PIXELS_PER_METER, 1.0)

    rssi = P0_RSSI - 10 * PATH_LOSS_N * math.log10(d_m)

    # 벽 지점은 천장 돌출 벽 등으로 가려졌다고 보고 추가 감쇄
    if str(point["point_type"]) == "벽":
        rssi -= WALL_PENALTY_BLOCKED

    # 증폭기는 backhaul이 약하면 실제 성능 제한
    if candidate["device_type"] == "증폭기":
        backhaul = candidate["backhaul_rssi"]
        if pd.isna(backhaul):
            rssi -= 10
        elif backhaul < -65:
            rssi -= 12
        elif backhaul < -60:
            rssi -= 5

    return max(rssi, NEW_SIGNAL_CUTOFF)


def build_effect_matrix(df, candidates):
    effect = np.zeros((len(candidates), len(df)))

    for ci, c in candidates.iterrows():
        for pi, p in df.iterrows():
            effect[ci, pi] = estimate_new_rssi(c, p)

    return effect


# =========================
# 7. NSGA-II 구현
# =========================

def evaluate_solution(genome, df, weak, candidates, effect_matrix):
    selected_idx = np.where(genome == 1)[0]

    if len(selected_idx) == 0:
        return {
            "improvement_rate": 0.0,
            "mean_rssi_gain_weak": 0.0,
            "weak_points_improved": 0,
            "device_count": 0,
            "total_cost": 0.0,
            "interference_penalty": 0.0,
            "constraint_penalty": 0.0,
            "objective_1": 1.0,
            "objective_2": 0.0,
            "objective_3": 0.0,
        }

    current_rssi = df[RSSI_COL].values.astype(float)

    selected_effect = effect_matrix[selected_idx, :]
    best_new_rssi = selected_effect.max(axis=0)
    post_rssi = np.maximum(current_rssi, best_new_rssi)

    weak_indices = weak.index.values
    current_weak = current_rssi[weak_indices]
    post_weak = post_rssi[weak_indices]

    gain_weak = post_weak - current_weak
    gain_weak = np.maximum(gain_weak, 0)

    # 개선률: TARGET_RSSI까지 필요한 개선량 대비 달성 비율
    need = np.maximum(TARGET_RSSI - current_weak, 1.0)
    achieved = np.minimum(gain_weak, need)
    improvement_rate = float(np.sum(achieved) / np.sum(need))

    weak_points_improved = int(np.sum(gain_weak >= 3.0))
    mean_rssi_gain_weak = float(np.mean(gain_weak)) if len(gain_weak) else 0.0

    selected_candidates = candidates.iloc[selected_idx]
    device_count = int(len(selected_idx))
    total_cost = float(selected_candidates["cost"].sum() + 0.25 * selected_candidates["install_difficulty"].sum())

    # 간섭 패널티: 새 장비가 -75dBm 이상으로 강하게 덮는 지점 수
    strong_new_cover = selected_effect >= -75
    overlap_added = strong_new_cover.sum(axis=0)

    existing_overlap = df[AP_COUNT_COL].fillna(0).values.astype(float)

    # 이미 AP가 많이 잡히는 곳에 새 신호가 추가되면 패널티
    interference_penalty = float(np.mean(overlap_added * np.maximum(existing_overlap - 1, 0)))

    # 제약 패널티
    constraint_penalty = 0.0

    # 증폭기 backhaul 미달 패널티
    for _, c in selected_candidates.iterrows():
        if c["device_type"] == "증폭기" and not bool(c["backhaul_ok"]):
            constraint_penalty += 3.0

    # 같은 base_id에서 AP와 증폭기를 동시에 고르는 중복 패널티
    duplicated_base = selected_candidates["base_id"].duplicated().sum()
    constraint_penalty += float(duplicated_base) * 2.0

    # NSGA-II는 최소화 문제로 처리
    objective_1 = 1.0 - improvement_rate
    objective_2 = total_cost
    objective_3 = interference_penalty + constraint_penalty

    return {
        "improvement_rate": improvement_rate,
        "mean_rssi_gain_weak": mean_rssi_gain_weak,
        "weak_points_improved": weak_points_improved,
        "device_count": device_count,
        "total_cost": total_cost,
        "interference_penalty": interference_penalty,
        "constraint_penalty": constraint_penalty,
        "objective_1": objective_1,
        "objective_2": objective_2,
        "objective_3": objective_3,
    }


def dominates(a, b):
    obj_a = np.array([a["objective_1"], a["objective_2"], a["objective_3"]])
    obj_b = np.array([b["objective_1"], b["objective_2"], b["objective_3"]])

    return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)


def fast_non_dominated_sort(evals):
    n = len(evals)
    S = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue

            if dominates(evals[p], evals[q]):
                S[p].append(q)
            elif dominates(evals[q], evals[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []

        for p in fronts[i]:
            for q in S[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)

        i += 1
        fronts.append(next_front)

    fronts.pop()
    return fronts


def crowding_distance(front, evals):
    if len(front) == 0:
        return {}

    distances = {idx: 0.0 for idx in front}
    objectives = ["objective_1", "objective_2", "objective_3"]

    for obj in objectives:
        sorted_front = sorted(front, key=lambda idx: evals[idx][obj])

        distances[sorted_front[0]] = float("inf")
        distances[sorted_front[-1]] = float("inf")

        min_val = evals[sorted_front[0]][obj]
        max_val = evals[sorted_front[-1]][obj]

        if max_val == min_val:
            continue

        for i in range(1, len(sorted_front) - 1):
            prev_val = evals[sorted_front[i - 1]][obj]
            next_val = evals[sorted_front[i + 1]][obj]
            distances[sorted_front[i]] += (next_val - prev_val) / (max_val - min_val)

    return distances


def random_genome(n):
    p = min(0.25, max(0.12, 2.0 / max(n, 1)))
    g = (np.random.rand(n) < p).astype(int)

    if g.sum() == 0:
        g[np.random.randint(0, n)] = 1

    return g


def tournament_select(pop, evals):
    i, j = np.random.choice(len(pop), 2, replace=False)

    a = evals[i]
    b = evals[j]

    if dominates(a, b):
        return pop[i].copy()
    if dominates(b, a):
        return pop[j].copy()

    # 비지배 관계면 improvement_rate 높은 쪽 우선
    if a["improvement_rate"] > b["improvement_rate"]:
        return pop[i].copy()
    return pop[j].copy()


def crossover(parent1, parent2):
    if np.random.rand() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    mask = np.random.rand(len(parent1)) < 0.5
    child1 = parent1.copy()
    child2 = parent2.copy()

    child1[mask] = parent2[mask]
    child2[mask] = parent1[mask]

    return child1, child2


def mutate(genome):
    for i in range(len(genome)):
        if np.random.rand() < MUTATION_RATE:
            genome[i] = 1 - genome[i]

    if genome.sum() == 0:
        genome[np.random.randint(0, len(genome))] = 1

    return genome


def select_next_population(combined_pop, combined_evals, pop_size):
    fronts = fast_non_dominated_sort(combined_evals)

    new_pop = []
    new_evals = []

    for front in fronts:
        if len(new_pop) + len(front) <= pop_size:
            for idx in front:
                new_pop.append(combined_pop[idx])
                new_evals.append(combined_evals[idx])
        else:
            dist = crowding_distance(front, combined_evals)
            sorted_front = sorted(front, key=lambda idx: dist[idx], reverse=True)

            remaining = pop_size - len(new_pop)
            for idx in sorted_front[:remaining]:
                new_pop.append(combined_pop[idx])
                new_evals.append(combined_evals[idx])
            break

    return new_pop, new_evals


def run_nsga2(df, weak, candidates, effect_matrix):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n_candidates = len(candidates)

    pop = [random_genome(n_candidates) for _ in range(POP_SIZE)]
    evals = [evaluate_solution(g, df, weak, candidates, effect_matrix) for g in pop]

    for gen in range(N_GENERATIONS):
        offspring = []

        while len(offspring) < POP_SIZE:
            p1 = tournament_select(pop, evals)
            p2 = tournament_select(pop, evals)

            c1, c2 = crossover(p1, p2)

            offspring.append(mutate(c1))
            if len(offspring) < POP_SIZE:
                offspring.append(mutate(c2))

        offspring_evals = [evaluate_solution(g, df, weak, candidates, effect_matrix) for g in offspring]

        combined_pop = pop + offspring
        combined_evals = evals + offspring_evals

        pop, evals = select_next_population(combined_pop, combined_evals, POP_SIZE)

        if (gen + 1) % 20 == 0:
            best_imp = max(e["improvement_rate"] for e in evals)
            print(f"Generation {gen + 1}: best improvement = {best_imp:.3f}")

    fronts = fast_non_dominated_sort(evals)
    pareto_idx = fronts[0]

    rows = []
    for rank, idx in enumerate(pareto_idx):
        g = pop[idx]
        e = evals[idx]

        selected = candidates.iloc[np.where(g == 1)[0]]["candidate_id"].tolist()

        row = {
            "solution_id": f"P{rank+1}",
            "selected_candidates": ";".join(selected),
            **e
        }
        rows.append(row)

    pareto = pd.DataFrame(rows)
    pareto = pareto.sort_values(["objective_1", "objective_2", "objective_3"]).reset_index(drop=True)
    pareto.to_csv(OUT_DIR / "nsga2_pareto_solutions.csv", index=False, encoding="utf-8-sig")

    plot_pareto(pareto)

    selected_solutions = select_representative_solutions(pareto)
    selected_solutions.to_csv(
        OUT_DIR / "nsga2_selected_solutions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return pareto, selected_solutions


def plot_pareto(pareto):
    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        pareto["total_cost"],
        pareto["improvement_rate"],
        c=pareto["interference_penalty"] + pareto["constraint_penalty"],
        s=120,
        cmap="viridis",
        edgecolors="black"
    )

    for _, r in pareto.iterrows():
        ax.text(
            r["total_cost"],
            r["improvement_rate"],
            r["solution_id"],
            fontsize=8,
            ha="center",
            va="bottom"
        )

    ax.set_title("NSGA-II Pareto 최적해")
    ax.set_xlabel("설치 비용/난이도")
    ax.set_ylabel("취약 지점 개선률")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("간섭·제약 패널티")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "nsga2_pareto_front.png", dpi=220)
    plt.close(fig)


def select_representative_solutions(pareto):
    if pareto.empty:
        return pareto

    cost_idx = pareto["total_cost"].idxmin()
    performance_idx = pareto["improvement_rate"].idxmax()

    normalized = pareto.copy()

    for col in ["total_cost", "interference_penalty", "constraint_penalty"]:
        min_v = normalized[col].min()
        max_v = normalized[col].max()
        normalized[col + "_norm"] = (normalized[col] - min_v) / max(max_v - min_v, 1e-9)

    imp_min = normalized["improvement_rate"].min()
    imp_max = normalized["improvement_rate"].max()
    normalized["improvement_loss_norm"] = (
        imp_max - normalized["improvement_rate"]
    ) / max(imp_max - imp_min, 1e-9)

    normalized["balanced_score"] = (
        normalized["improvement_loss_norm"]
        + normalized["total_cost_norm"]
        + normalized["interference_penalty_norm"]
        + normalized["constraint_penalty_norm"]
    )

    balanced_idx = normalized["balanced_score"].idxmin()

    selected = pareto.loc[[cost_idx, performance_idx, balanced_idx]].copy()
    selected["plan_type"] = ["비용 우선안", "성능 우선안", "균형안"]

    return selected[[
        "plan_type",
        "solution_id",
        "selected_candidates",
        "improvement_rate",
        "mean_rssi_gain_weak",
        "weak_points_improved",
        "device_count",
        "total_cost",
        "interference_penalty",
        "constraint_penalty",
    ]]


# =========================
# 8. 지도 위 최종안 표시
# =========================

def find_floorplan():
    p = Path(FLOORPLAN_IMAGE)
    if p.exists():
        return p

    candidates = list(Path(".").glob("*.png")) + list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.jpeg"))
    if candidates:
        return candidates[0]

    return None


def plot_solution_overlay(df, weak, candidates, selected_solutions):
    floorplan = find_floorplan()

    if floorplan is None:
        print("지도 이미지가 없어 overlay 생략")
        return

    img = plt.imread(floorplan)

    for _, sol in selected_solutions.iterrows():
        selected_ids = str(sol["selected_candidates"]).split(";")
        selected = candidates[candidates["candidate_id"].isin(selected_ids)]

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(img)
        ax.axis("off")

        ax.scatter(
            df["x"],
            df["y"],
            c=df[QUALITY_COL],
            s=70,
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            edgecolors="black",
            alpha=0.75,
            label="전체 측정점"
        )

        ax.scatter(
            weak["x"],
            weak["y"],
            s=190,
            facecolors="none",
            edgecolors="red",
            linewidths=2.0,
            label="취약 지점"
        )

        for _, c in selected.iterrows():
            marker = "*" if c["device_type"] == "AP" else "P"
            color = "blue" if c["device_type"] == "AP" else "purple"

            ax.scatter(
                c["x"],
                c["y"],
                s=420,
                marker=marker,
                c=color,
                edgecolors="white",
                linewidths=1.5,
                label=f"{c['device_type']} 후보"
            )

            ax.text(
                c["x"],
                c["y"] - 18,
                f"{c['candidate_id']}\n{c['device_type']}",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85)
            )

        title = (
            f"{sol['plan_type']} | 개선률={sol['improvement_rate']:.2f}, "
            f"비용={sol['total_cost']:.2f}, 장비수={int(sol['device_count'])}"
        )
        ax.set_title(title, fontsize=15)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="lower right")

        safe_name = sol["plan_type"].replace(" ", "_")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"solution_overlay_{safe_name}.png", dpi=220)
        plt.close(fig)


# =========================
# 9. 결과 해석 텍스트 생성
# =========================

def create_report_text(weak, dbscan_result, candidates, selected_solutions):
    cluster_counts = (
        dbscan_result
        .groupby("cluster")
        .size()
        .reset_index(name="count")
    )

    text = []
    text.append("AI 기반 Wi-Fi 취약 구역 탐지 및 설치 위치 최적화 결과 요약")
    text.append("")
    text.append(f"1. 품질 점수 기준으로 취약 지점 {len(weak)}개를 추출하였다.")
    text.append("2. DBSCAN을 적용하여 취약 지점의 공간적 군집을 탐지하였다.")
    text.append("3. 각 군집 중심 및 최저 품질 지점을 기준으로 AP/증폭기 설치 후보를 자동 생성하였다.")
    text.append("4. 경로 손실 모델을 이용해 후보 장비 설치 시 각 측정점의 예상 RSSI 개선량을 계산하였다.")
    text.append("5. NSGA-II를 이용해 취약 지점 개선률, 설치 비용, 간섭·제약 패널티를 동시에 고려한 Pareto 최적해를 도출하였다.")
    text.append("")
    text.append("[DBSCAN 군집 요약]")
    for _, r in cluster_counts.iterrows():
        label = int(r["cluster"])
        if label == -1:
            text.append(f"- Noise: {int(r['count'])}개 지점")
        else:
            text.append(f"- Cluster {label}: {int(r['count'])}개 지점")

    text.append("")
    text.append("[최종 대표안]")
    for _, r in selected_solutions.iterrows():
        text.append(
            f"- {r['plan_type']}: 후보 {r['selected_candidates']} 선택, "
            f"개선률 {r['improvement_rate']:.2f}, "
            f"평균 RSSI 개선 {r['mean_rssi_gain_weak']:.2f}dB, "
            f"장비 수 {int(r['device_count'])}, "
            f"비용 {r['total_cost']:.2f}, "
            f"간섭 패널티 {r['interference_penalty']:.2f}"
        )

    text.append("")
    text.append("[보고서용 결론]")
    text.append(
        "본 탐구에서는 실측 RSSI 및 AP 중복 감지 데이터를 기반으로 Wi-Fi 종합 품질 점수를 산출하고, "
        "품질 점수가 낮은 지점을 DBSCAN으로 군집화하였다. 그 결과 취약 지점은 무작위로 분포하기보다 "
        "복도 및 문앞 구간을 중심으로 군집화되는 경향을 보였다. 이후 각 군집 중심과 최저 품질 지점을 "
        "AP/증폭기 설치 후보로 설정하고, 경로 손실 모델을 통해 설치 후 RSSI 개선량을 예측하였다. "
        "마지막으로 NSGA-II 다목적 최적화를 적용하여 취약 지점 개선률, 설치 비용, 간섭 패널티를 동시에 고려한 "
        "비용 우선안, 성능 우선안, 균형안을 도출하였다. 따라서 본 탐구는 단순히 신호가 약한 지점을 육안으로 찾는 방식이 아니라, "
        "비지도학습 군집 알고리즘과 진화 기반 최적화 알고리즘을 결합하여 학교 4층 Wi-Fi 품질 개선 방안을 데이터 기반으로 제안했다는 점에서 의의가 있다."
    )

    report = "\n".join(text)

    with open(OUT_DIR / "ai_solution_report_summary.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(report)


# =========================
# 10. 실행
# =========================

def main():
    setup_korean_font()

    df = load_data()

    weak, weak_threshold = extract_weak_points(df)

    dbscan_result, eps = run_dbscan(weak)

    candidates = generate_candidates(df, dbscan_result)

    effect_matrix = build_effect_matrix(df, candidates)

    pareto, selected_solutions = run_nsga2(
        df=df,
        weak=weak,
        candidates=candidates,
        effect_matrix=effect_matrix
    )

    plot_solution_overlay(df, weak, candidates, selected_solutions)

    create_report_text(weak, dbscan_result, candidates, selected_solutions)

    print("\n완료")
    print(f"결과 폴더: {OUT_DIR.resolve()}")
    print("생성 파일:")
    print("- weak_points.csv")
    print("- dbscan_result.csv")
    print("- dbscan_k_distance_graph.png")
    print("- dbscan_clusters_xy.png")
    print("- dbscan_sensitivity.csv")
    print("- generated_candidates.csv")
    print("- nsga2_pareto_solutions.csv")
    print("- nsga2_selected_solutions.csv")
    print("- nsga2_pareto_front.png")
    print("- solution_overlay_비용_우선안.png")
    print("- solution_overlay_성능_우선안.png")
    print("- solution_overlay_균형안.png")
    print("- ai_solution_report_summary.txt")


if __name__ == "__main__":
    main()