# classroom_wifi_congestion_ai_v2.py
# 목적:
# 학생별 종합시간표 엑셀을 이용해 교실별 예상 Wi-Fi 혼잡도를 계산하고,
# K-means 알고리즘으로 교실을 혼잡 유형별로 분류한다.
#
# 주요 수정:
# 1. 운동장(프3층)은 Wi-Fi 분석 대상에서 제외
# 2. 교실별 종합 혼잡도뿐 아니라 교시별 혼잡도 그래프도 생성
# 3. txt 파일에는 결과 사진과 CSV 파일 설명만 작성
#
# 필요 패키지:
# pip install pandas numpy matplotlib openpyxl
#
# 실행:
# python classroom_wifi_congestion_ai_v2.py

import re
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

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "classroom_wifi_congestion_outputs"
OUT_DIR.mkdir(exist_ok=True)

MANUAL_FILES = {
    "시간표": "대신고_3학년_종합시간표_완벽마스터.xlsx",
    "수강인원": "대신고_3학년_과목별_수강인원_완성형.xlsx",
}

# True로 바꾸면 4층 교실만 분석
FOURTH_FLOOR_ONLY = False

# Wi-Fi 분석에서 제외할 장소
NO_WIFI_LOCATIONS = {
    "운동장",
    "운동장(프3층)",
}

# 혼잡 기준
BUSY_STUDENT_THRESHOLD = 30
VERY_BUSY_STUDENT_THRESHOLD = 40

# K-means 설정
K = 3
RANDOM_SEED = 42
MAX_ITER = 300


# =========================
# 1. 교실 이름 → 실제 호실 매핑
# =========================

CLASSROOM_TO_REAL_ROOM = {
    "3-1": "306",
    "3-2": "305",
    "3-3": "304",
    "3-4": "303",
    "3-5": "408",
    "3-6": "407",
    "3-7": "406",
    "3-8": "405",
    "3-9": "404",
    "3-10": "403",
    "글로벌실": "402",
    "미디어실": "401",
}


# =========================
# 2. 한글 폰트 설정
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
# 3. 파일 탐색
# =========================

def normalize_text(s):
    return unicodedata.normalize("NFC", str(s))


def find_excel_file(file_type):
    manual_name = MANUAL_FILES.get(file_type)

    search_roots = [
        SCRIPT_DIR,
        Path.cwd(),
        SCRIPT_DIR.parent,
    ]

    if manual_name:
        for root in search_roots:
            p = root / manual_name
            if p.exists():
                return p

    if file_type == "시간표":
        keywords = ["종합시간표", "시간표", "완벽마스터"]
    else:
        keywords = ["수강인원", "과목별", "완성형"]

    candidates = []

    for root in search_roots:
        if not root.exists():
            continue

        try:
            files = list(root.rglob("*.xlsx"))
        except Exception:
            files = []

        for p in files:
            name = normalize_text(p.name)

            if name.startswith("~$"):
                continue

            score = 0
            for kw in keywords:
                if normalize_text(kw) in name:
                    score += 1

            if score > 0:
                candidates.append((score, p.stat().st_mtime, p))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]

    print("\n찾은 xlsx 파일 목록:")
    all_xlsx = []
    for root in search_roots:
        if root.exists():
            try:
                all_xlsx.extend(list(root.rglob("*.xlsx")))
            except Exception:
                pass

    for p in all_xlsx:
        print("-", p)

    raise FileNotFoundError(
        f"{file_type} 엑셀 파일을 찾지 못했습니다.\n"
        f"엑셀 파일을 이 코드와 같은 폴더에 넣거나 MANUAL_FILES의 파일명을 실제 파일명으로 수정하세요."
    )


# =========================
# 4. 교실 위치 추출
# =========================

def normalize_location(location):
    if location is None or pd.isna(location):
        return None

    text = normalize_text(location).strip()
    text = text.replace(" ", "")

    if not text:
        return None

    m = re.match(r"3-(\d+)$", text)
    if m:
        return f"3-{int(m.group(1))}"

    m = re.match(r"2-(\d+)$", text)
    if m:
        return f"2-{int(m.group(1))}"

    special_map = {
        "글로벌실": "글로벌실",
        "미디어실": "미디어실",
        "과학실": "과학실",
        "진로실": "진로실",
        "자주학1": "자주학1",
        "자주학2": "자주학2",
        "자주학3": "자주학3",
        "자주학4": "자주학4",
        "자주학5": "자주학5",
        "인공지능실": "인공지능실",
        "운동장": "운동장(프3층)",
        "운동장(프3층)": "운동장(프3층)",
        "3-2이선복": "3-2이선복",
        "3-2최슬기": "3-2최슬기",
    }

    return special_map.get(text, text)


def extract_location_from_cell(cell_value):
    """
    예:
    미적분(F)(교사명) [3-10] -> 3-10
    영어독해 [글로벌실] -> 글로벌실
    """
    if cell_value is None or pd.isna(cell_value):
        return None

    text = normalize_text(cell_value).strip()

    if not text:
        return None

    matches = re.findall(r"\[([^\]]+)\]", text)
    if not matches:
        return None

    location = matches[-1].strip()
    return normalize_location(location)


def location_to_real_room(location):
    return CLASSROOM_TO_REAL_ROOM.get(location, "unknown")


def is_4th_floor_room(real_room):
    return str(real_room).startswith("4")


# =========================
# 5. 정렬 함수
# =========================

def period_sort_key(period):
    text = str(period)

    day_order = {
        "월": 0,
        "화": 1,
        "수": 2,
        "목": 3,
        "금": 4,
    }

    m = re.match(r"^([월화수목금])_(\d+)교시$", text)
    if m:
        return (day_order.get(m.group(1), 99), int(m.group(2)))

    return (99, text)


def classroom_sort_key(x):
    x = str(x)

    m = re.match(r"3-(\d+)$", x)
    if m:
        return (0, int(m.group(1)))

    m = re.match(r"2-(\d+)$", x)
    if m:
        return (1, int(m.group(1)))

    order = {
        "미디어실": 100,
        "글로벌실": 101,
        "과학실": 102,
        "인공지능실": 103,
        "진로실": 104,
        "자주학1": 105,
        "자주학2": 106,
        "자주학3": 107,
        "자주학4": 108,
        "자주학5": 109,
        "3-2이선복": 110,
        "3-2최슬기": 111,
    }

    return (2, order.get(x, 999), x)


# =========================
# 6. 시간표 엑셀 로드
# =========================

def detect_timetable_sheet(excel_path):
    xls = pd.ExcelFile(excel_path)

    preferred = [
        "text_학생별_시간표",
        "학생별_시간표",
        "시간표",
    ]

    for s in preferred:
        if s in xls.sheet_names:
            return s

    for s in xls.sheet_names:
        temp = pd.read_excel(excel_path, sheet_name=s, nrows=5)
        cols = [normalize_text(c).strip() for c in temp.columns]
        if "학반" in cols and "번호" in cols:
            return s

    return xls.sheet_names[0]


def load_timetable():
    path = find_excel_file("시간표")
    print(f"시간표 파일 사용: {path}")

    sheet_name = detect_timetable_sheet(path)
    print(f"시간표 시트 사용: {sheet_name}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [normalize_text(c).strip() for c in df.columns]

    required_base = ["학반", "번호"]
    for col in required_base:
        if col not in df.columns:
            raise ValueError(f"시간표 파일에 '{col}' 열이 없습니다. 현재 열: {list(df.columns)}")

    period_cols = [
        c for c in df.columns
        if re.match(r"^[월화수목금]_\d+교시$", str(c))
    ]

    if not period_cols:
        renamed = {}
        for c in df.columns:
            m = re.match(r"^([월화수목금])\s*[_-]?\s*(\d+)\s*교시$", str(c))
            if m:
                renamed[c] = f"{m.group(1)}_{m.group(2)}교시"

        if renamed:
            df = df.rename(columns=renamed)
            period_cols = list(renamed.values())

    if not period_cols:
        raise ValueError(
            "교시 열을 찾지 못했습니다. 예: 월_1교시, 화_2교시\n"
            f"현재 열: {list(df.columns)}"
        )

    period_cols = sorted(period_cols, key=period_sort_key)

    print(f"학생 수: {len(df)}")
    print(f"교시 열 수: {len(period_cols)}")

    return df, period_cols, path


# =========================
# 7. 학생별 위치 long table 생성
# =========================

def build_student_location_long(timetable_df, period_cols):
    rows = []
    excluded_rows = []

    for idx, student in timetable_df.iterrows():
        학반 = student.get("학반", "")
        번호 = student.get("번호", "")
        이름 = student.get("이름", "")
        student_id = f"{학반}-{번호}"

        for period in period_cols:
            raw_cell = student.get(period, None)
            classroom = extract_location_from_cell(raw_cell)

            if classroom is None:
                continue

            if classroom in NO_WIFI_LOCATIONS:
                excluded_rows.append({
                    "student_id": student_id,
                    "학반": 학반,
                    "번호": 번호,
                    "이름": 이름,
                    "period": period,
                    "excluded_location": classroom,
                    "reason": "운동장에는 실내 Wi-Fi AP가 없으므로 분석에서 제외",
                    "raw_subject_cell": raw_cell,
                })
                continue

            real_room = location_to_real_room(classroom)
            is_4f = is_4th_floor_room(real_room)

            if FOURTH_FLOOR_ONLY and not is_4f:
                continue

            rows.append({
                "student_id": student_id,
                "학반": 학반,
                "번호": 번호,
                "이름": 이름,
                "period": period,
                "classroom_name": classroom,
                "real_room": real_room,
                "is_4th_floor": is_4f,
                "raw_subject_cell": raw_cell,
            })

    long_df = pd.DataFrame(rows)

    if long_df.empty:
        raise ValueError("시간표에서 [교실] 정보를 추출하지 못했습니다.")

    long_df.to_csv(
        OUT_DIR / "student_period_location_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    excluded_df = pd.DataFrame(excluded_rows)
    excluded_df.to_csv(
        OUT_DIR / "excluded_no_wifi_locations.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return long_df, excluded_df


# =========================
# 8. 교실별·교시별 부하 계산
# =========================

def build_classroom_period_load(long_df):
    load = (
        long_df
        .groupby(["period", "classroom_name", "real_room", "is_4th_floor"], as_index=False)
        .agg(
            estimated_students=("student_id", "nunique")
        )
    )

    load["period"] = pd.Categorical(
        load["period"],
        categories=sorted(load["period"].unique(), key=period_sort_key),
        ordered=True
    )

    load = load.sort_values(["period", "estimated_students"], ascending=[True, False])

    load.to_csv(
        OUT_DIR / "classroom_period_load.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return load


# =========================
# 9. 교실별 feature 생성
# =========================

def build_classroom_features(load):
    all_periods = sorted(load["period"].astype(str).unique(), key=period_sort_key)

    rows = []

    for classroom in sorted(load["classroom_name"].unique(), key=classroom_sort_key):
        part = load[load["classroom_name"] == classroom].copy()

        real_room = part["real_room"].iloc[0]
        is_4f = bool(part["is_4th_floor"].iloc[0])

        period_to_count = dict(zip(part["period"].astype(str), part["estimated_students"]))
        counts = np.array([period_to_count.get(p, 0) for p in all_periods], dtype=float)

        mean_students = float(np.mean(counts))
        max_students = float(np.max(counts))
        p90_students = float(np.percentile(counts, 90))
        std_students = float(np.std(counts))
        total_student_periods = int(np.sum(counts))

        busy_periods = int(np.sum(counts >= BUSY_STUDENT_THRESHOLD))
        very_busy_periods = int(np.sum(counts >= VERY_BUSY_STUDENT_THRESHOLD))

        congestion_ratio = float(busy_periods / len(counts))
        very_congestion_ratio = float(very_busy_periods / len(counts))

        top_periods = sorted(
            [(p, int(period_to_count.get(p, 0))) for p in all_periods],
            key=lambda x: x[1],
            reverse=True
        )[:6]

        top_periods_text = "; ".join([f"{p}:{c}명" for p, c in top_periods if c > 0])

        rows.append({
            "classroom_name": classroom,
            "real_room": real_room,
            "is_4th_floor": is_4f,
            "mean_students": mean_students,
            "max_students": max_students,
            "p90_students": p90_students,
            "std_students": std_students,
            "busy_periods_ge_30": busy_periods,
            "very_busy_periods_ge_40": very_busy_periods,
            "congestion_ratio_ge_30": congestion_ratio,
            "very_congestion_ratio_ge_40": very_congestion_ratio,
            "total_student_periods": total_student_periods,
            "top_periods": top_periods_text,
        })

    features = pd.DataFrame(rows)
    features = add_congestion_score(features)

    features.to_csv(
        OUT_DIR / "classroom_congestion_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return features


def add_congestion_score(features):
    df = features.copy()

    def minmax(col):
        s = df[col].astype(float)
        if s.max() == s.min():
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    df["mean_norm"] = minmax("mean_students")
    df["max_norm"] = minmax("max_students")
    df["p90_norm"] = minmax("p90_students")
    df["ratio_norm"] = minmax("congestion_ratio_ge_30")
    df["std_norm"] = minmax("std_students")

    df["congestion_score"] = (
        0.30 * df["mean_norm"]
        + 0.30 * df["max_norm"]
        + 0.20 * df["p90_norm"]
        + 0.15 * df["ratio_norm"]
        + 0.05 * df["std_norm"]
    ) * 100

    return df


# =========================
# 10. K-means 직접 구현
# =========================

def standardize(X):
    X = X.astype(float)

    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0

    X_filled = np.where(np.isnan(X), mean, X)
    Z = (X_filled - mean) / std

    return Z


def init_centroids_farthest(Z, k):
    np.random.seed(RANDOM_SEED)

    n = len(Z)
    first = np.random.randint(0, n)
    centroids = [Z[first]]

    while len(centroids) < k:
        distances = []

        for z in Z:
            d = min(np.linalg.norm(z - c) for c in centroids)
            distances.append(d)

        next_idx = int(np.argmax(distances))
        centroids.append(Z[next_idx])

    return np.array(centroids)


def run_kmeans(Z, k=3, max_iter=300):
    n = len(Z)
    k = min(k, n)

    centroids = init_centroids_farthest(Z, k)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        old_labels = labels.copy()

        dist = np.array([
            [np.linalg.norm(z - c) for c in centroids]
            for z in Z
        ])

        labels = np.argmin(dist, axis=1)

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


def classify_classrooms_kmeans(features):
    feature_cols = [
        "mean_students",
        "max_students",
        "p90_students",
        "congestion_ratio_ge_30",
        "std_students",
    ]

    X = features[feature_cols].values
    Z = standardize(X)

    labels, centroids, inertia = run_kmeans(Z, k=K, max_iter=MAX_ITER)

    result = features.copy()
    result["cluster"] = labels

    cluster_summary = (
        result
        .groupby("cluster", as_index=False)
        .agg(
            n_classrooms=("classroom_name", "count"),
            mean_congestion_score=("congestion_score", "mean"),
            mean_students=("mean_students", "mean"),
            mean_max_students=("max_students", "mean"),
            mean_busy_ratio=("congestion_ratio_ge_30", "mean"),
        )
    )

    sorted_clusters = cluster_summary.sort_values("mean_congestion_score")

    low_cluster = sorted_clusters.iloc[0]["cluster"]
    high_cluster = sorted_clusters.iloc[-1]["cluster"]

    type_map = {}

    for c in cluster_summary["cluster"]:
        if c == high_cluster:
            type_map[c] = "혼잡 병목형"
        elif c == low_cluster:
            type_map[c] = "저부하 안정형"
        else:
            type_map[c] = "중간 부하형"

    result["congestion_type"] = result["cluster"].map(type_map)
    cluster_summary["congestion_type"] = cluster_summary["cluster"].map(type_map)

    result = result.sort_values("congestion_score", ascending=False).reset_index(drop=True)

    result.to_csv(
        OUT_DIR / "classroom_kmeans_result.csv",
        index=False,
        encoding="utf-8-sig"
    )

    cluster_summary.to_csv(
        OUT_DIR / "classroom_kmeans_cluster_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    elbow_rows = []
    max_k = min(8, len(features))

    for k in range(1, max_k + 1):
        labels_k, centroids_k, inertia_k = run_kmeans(Z, k=k, max_iter=MAX_ITER)
        elbow_rows.append({
            "k": k,
            "inertia": inertia_k,
        })

    elbow_df = pd.DataFrame(elbow_rows)
    elbow_df.to_csv(
        OUT_DIR / "kmeans_elbow.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return result, cluster_summary, elbow_df


# =========================
# 11. 수강인원 파일 보조 분석
# =========================

def detect_course_count_sheet(excel_path):
    xls = pd.ExcelFile(excel_path)

    preferred = [
        "수업별_수강인원",
        "과목별_수강인원",
        "수강인원",
    ]

    for s in preferred:
        if s in xls.sheet_names:
            return s

    return xls.sheet_names[0]


def analyze_course_count_file():
    try:
        path = find_excel_file("수강인원")
    except FileNotFoundError:
        print("수강인원 파일을 찾지 못해 보조 분석은 생략합니다.")
        return None

    print(f"수강인원 파일 사용: {path}")

    sheet_name = detect_course_count_sheet(path)
    print(f"수강인원 시트 사용: {sheet_name}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [normalize_text(c).strip() for c in df.columns]

    place_col = None
    count_col = None
    subject_col = None

    for c in df.columns:
        if "장소" in c or "교실" in c:
            place_col = c
        if "수강" in c and ("수" in c or "인원" in c):
            count_col = c
        if "과목" in c:
            subject_col = c

    if place_col is None or count_col is None:
        print("수강인원 파일에서 장소/인원 열을 찾지 못해 보조 분석 생략")
        print("현재 열:", list(df.columns))
        return None

    df["classroom_name"] = df[place_col].apply(normalize_location)
    df = df[~df["classroom_name"].isin(NO_WIFI_LOCATIONS)].copy()

    df["real_room"] = df["classroom_name"].apply(location_to_real_room)
    df["is_4th_floor"] = df["real_room"].astype(str).str.startswith("4")
    df["course_students"] = pd.to_numeric(df[count_col], errors="coerce")

    if FOURTH_FLOOR_ONLY:
        df = df[df["is_4th_floor"]].copy()

    summary = (
        df
        .dropna(subset=["classroom_name", "course_students"])
        .groupby(["classroom_name", "real_room", "is_4th_floor"], as_index=False)
        .agg(
            course_count=(subject_col if subject_col else count_col, "count"),
            mean_course_size=("course_students", "mean"),
            max_course_size=("course_students", "max"),
            total_course_students=("course_students", "sum"),
        )
        .sort_values("max_course_size", ascending=False)
    )

    df.to_csv(
        OUT_DIR / "course_count_raw_with_classroom.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        OUT_DIR / "course_count_classroom_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return summary


# =========================
# 12. 시각화
# =========================

def plot_period_heatmap(load):
    pivot = load.pivot_table(
        index="classroom_name",
        columns="period",
        values="estimated_students",
        aggfunc="sum",
        fill_value=0
    )

    pivot = pivot.loc[sorted(pivot.index, key=classroom_sort_key)]
    pivot = pivot[sorted(pivot.columns, key=period_sort_key)]

    fig, ax = plt.subplots(figsize=(18, 8))

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_title("교시별·교실별 예상 학생 밀집도")
    ax.set_xlabel("교시")
    ax.set_ylabel("교실")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = int(pivot.iloc[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=6)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("예상 학생 수")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "classroom_period_load_heatmap.png", dpi=220)
    plt.close(fig)


def plot_congestion_bar(result):
    data = result.sort_values("congestion_score", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.barh(data["classroom_name"], data["congestion_score"])
    ax.set_title("교실별 Wi-Fi 예상 혼잡도 점수")
    ax.set_xlabel("혼잡도 점수")
    ax.set_ylabel("교실")

    for i, (_, r) in enumerate(data.iterrows()):
        ax.text(
            r["congestion_score"] + 1,
            i,
            f"{r['congestion_type']} / 최대 {int(r['max_students'])}명",
            va="center",
            fontsize=8
        )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "classroom_congestion_score_bar.png", dpi=220)
    plt.close(fig)


def plot_kmeans_scatter(result):
    fig, ax = plt.subplots(figsize=(9, 6))

    order = ["저부하 안정형", "중간 부하형", "혼잡 병목형"]

    for t in order:
        part = result[result["congestion_type"] == t]

        if part.empty:
            continue

        ax.scatter(
            part["mean_students"],
            part["max_students"],
            s=180,
            edgecolors="black",
            label=t
        )

        for _, r in part.iterrows():
            ax.text(
                r["mean_students"],
                r["max_students"] + 0.8,
                r["classroom_name"],
                ha="center",
                fontsize=8
            )

    ax.axhline(BUSY_STUDENT_THRESHOLD, linestyle="--", linewidth=1)
    ax.set_title("K-means 기반 교실 Wi-Fi 혼잡 유형 분류")
    ax.set_xlabel("평균 예상 학생 수")
    ax.set_ylabel("최대 예상 학생 수")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "classroom_kmeans_scatter.png", dpi=220)
    plt.close(fig)


def plot_elbow(elbow_df):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
    ax.set_title("K-means K값 선택 참고 Elbow Graph")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "kmeans_elbow_graph.png", dpi=220)
    plt.close(fig)


def plot_each_period_top_classrooms(load, top_n=10):
    period_dir = OUT_DIR / "period_top_classrooms"
    period_dir.mkdir(exist_ok=True)

    periods = sorted(load["period"].astype(str).unique(), key=period_sort_key)

    summary_rows = []

    for period in periods:
        part = load[load["period"].astype(str) == period].copy()
        part = part.sort_values("estimated_students", ascending=False).head(top_n)

        if part.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        data = part.sort_values("estimated_students", ascending=True)

        ax.barh(data["classroom_name"], data["estimated_students"])
        ax.set_title(f"{period} 교실별 예상 Wi-Fi 접속 부하 TOP {top_n}")
        ax.set_xlabel("예상 학생 수")
        ax.set_ylabel("교실")

        for i, (_, r) in enumerate(data.iterrows()):
            ax.text(
                r["estimated_students"] + 0.5,
                i,
                f"{int(r['estimated_students'])}명",
                va="center",
                fontsize=9
            )

        fig.tight_layout()
        safe_period = period.replace("/", "_").replace(" ", "")
        fig.savefig(period_dir / f"{safe_period}_top_classrooms.png", dpi=220)
        plt.close(fig)

        for _, r in part.iterrows():
            summary_rows.append({
                "period": period,
                "classroom_name": r["classroom_name"],
                "real_room": r["real_room"],
                "estimated_students": r["estimated_students"],
            })

    period_top_df = pd.DataFrame(summary_rows)
    period_top_df.to_csv(
        OUT_DIR / "period_top_classrooms_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


def plot_global_period_classroom_peak(load, top_n=20):
    data = load.copy()
    data = data.sort_values("estimated_students", ascending=False).head(top_n)

    data["label"] = data["period"].astype(str) + " / " + data["classroom_name"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 8))

    plot_data = data.sort_values("estimated_students", ascending=True)

    ax.barh(plot_data["label"], plot_data["estimated_students"])
    ax.set_title(f"전체 교시-교실 조합 중 예상 Wi-Fi 접속 부하 TOP {top_n}")
    ax.set_xlabel("예상 학생 수")
    ax.set_ylabel("교시 / 교실")

    for i, (_, r) in enumerate(plot_data.iterrows()):
        ax.text(
            r["estimated_students"] + 0.5,
            i,
            f"{int(r['estimated_students'])}명",
            va="center",
            fontsize=9
        )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "global_period_classroom_peak_top20.png", dpi=220)
    plt.close(fig)

    data.to_csv(
        OUT_DIR / "global_period_classroom_peak_top20.csv",
        index=False,
        encoding="utf-8-sig"
    )


# =========================
# 13. 추천안
# =========================

def create_recommendations(result):
    rows = []

    for _, r in result.iterrows():
        if r["congestion_type"] == "혼잡 병목형":
            priority = "상"
            action = "유선 AP 추가 설치, 고밀도 수업 시간대 부하 분산, 채널 재배치 우선 검토"
        elif r["max_students"] >= VERY_BUSY_STUDENT_THRESHOLD:
            priority = "상"
            action = "특정 교시에 학생 수가 급증하므로 보조 AP 또는 사용 분산 검토"
        elif r["max_students"] >= BUSY_STUDENT_THRESHOLD:
            priority = "중"
            action = "혼잡 시간대 ping 재측정 및 AP 설정 점검"
        else:
            priority = "하"
            action = "우선 장비 추가 필요성 낮음"

        rows.append({
            "classroom_name": r["classroom_name"],
            "real_room": r["real_room"],
            "is_4th_floor": r["is_4th_floor"],
            "congestion_type": r["congestion_type"],
            "congestion_score": r["congestion_score"],
            "mean_students": r["mean_students"],
            "max_students": r["max_students"],
            "busy_periods_ge_30": r["busy_periods_ge_30"],
            "very_busy_periods_ge_40": r["very_busy_periods_ge_40"],
            "top_periods": r["top_periods"],
            "priority": priority,
            "recommended_action": action,
        })

    rec = pd.DataFrame(rows)

    priority_order = {"상": 0, "중": 1, "하": 2}
    rec["priority_order"] = rec["priority"].map(priority_order)
    rec = rec.sort_values(
        ["priority_order", "congestion_score"],
        ascending=[True, False]
    ).drop(columns=["priority_order"])

    rec.to_csv(
        OUT_DIR / "classroom_wifi_recommendations.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return rec


# =========================
# 14. TXT 설명 파일 생성
# =========================

def create_output_file_explanation_txt():
    lines = []

    lines.append("결과 파일 설명")
    lines.append("")
    lines.append("이 txt 파일은 생성된 CSV 파일과 사진 파일이 무엇을 의미하는지 설명하기 위한 파일이다.")
    lines.append("운동장(프3층)은 실내 Wi-Fi AP 분석 대상이 아니므로 결과 계산에서 제외하였다.")
    lines.append("")

    lines.append("[CSV 파일 설명]")
    lines.append("")
    lines.append("1. student_period_location_long.csv")
    lines.append("- 학생 한 명이 특정 요일·교시에 어느 교실에 있는지를 길게 정리한 원본 분석표이다.")
    lines.append("- 예를 들어 한 학생이 월_1교시에 3-10에 있으면, 그 학생이 월_1교시에 3-10 교실의 Wi-Fi를 사용할 가능성이 있다고 본다.")
    lines.append("")

    lines.append("2. excluded_no_wifi_locations.csv")
    lines.append("- Wi-Fi 혼잡도 분석에서 제외한 장소 목록이다.")
    lines.append("- 운동장(프3층)은 실내 Wi-Fi AP가 없으므로 제외하였다.")
    lines.append("")

    lines.append("3. classroom_period_load.csv")
    lines.append("- 각 교시마다 각 교실에 몇 명이 있는지 계산한 표이다.")
    lines.append("- 이 파일이 교시별 혼잡도 분석의 핵심 자료이다.")
    lines.append("- estimated_students 값이 클수록 해당 교시의 해당 교실에서 Wi-Fi 접속자가 많을 가능성이 크다.")
    lines.append("")

    lines.append("4. classroom_congestion_features.csv")
    lines.append("- 교실별 혼잡도 계산에 사용한 특징값을 정리한 표이다.")
    lines.append("- mean_students는 평균 학생 수, max_students는 가장 많이 몰린 순간의 학생 수를 의미한다.")
    lines.append("- congestion_score는 평균 학생 수, 최대 학생 수, 혼잡 교시 비율 등을 종합한 점수이다.")
    lines.append("")

    lines.append("5. classroom_kmeans_result.csv")
    lines.append("- K-means 알고리즘으로 각 교실을 저부하 안정형, 중간 부하형, 혼잡 병목형으로 분류한 결과이다.")
    lines.append("- 혼잡 병목형은 특정 시간대에 학생이 많이 몰려 Wi-Fi 지연이 발생할 가능성이 큰 교실이다.")
    lines.append("")

    lines.append("6. classroom_kmeans_cluster_summary.csv")
    lines.append("- K-means로 나뉜 각 군집의 평균 특징을 요약한 표이다.")
    lines.append("- 각 군집이 왜 저부하 안정형, 중간 부하형, 혼잡 병목형으로 해석되는지 확인할 수 있다.")
    lines.append("")

    lines.append("7. classroom_wifi_recommendations.csv")
    lines.append("- 교실별 개선 우선순위와 추천 조치를 정리한 표이다.")
    lines.append("- 우선순위가 상인 교실은 AP 추가 설치, 채널 재배치, 부하 분산 등을 먼저 검토할 필요가 있다.")
    lines.append("")

    lines.append("8. period_top_classrooms_summary.csv")
    lines.append("- 각 교시마다 학생 수가 많은 상위 교실을 정리한 표이다.")
    lines.append("- 어느 요일, 몇 교시에 특정 교실로 학생이 몰리는지 확인할 수 있다.")
    lines.append("")

    lines.append("9. global_period_classroom_peak_top20.csv")
    lines.append("- 전체 시간표 중 가장 학생 수가 많이 몰린 교시-교실 조합 상위 20개를 정리한 표이다.")
    lines.append("- Wi-Fi가 느려질 가능성이 큰 순간을 찾는 데 사용한다.")
    lines.append("")

    lines.append("10. course_count_raw_with_classroom.csv")
    lines.append("- 수강인원 파일에 있던 수업 정보를 교실 이름과 연결한 표이다.")
    lines.append("- 수업별로 어느 교실에서 몇 명이 수업을 듣는지 확인할 수 있다.")
    lines.append("")

    lines.append("11. course_count_classroom_summary.csv")
    lines.append("- 수강인원 파일을 교실별로 요약한 표이다.")
    lines.append("- 교실별 개설 수업 수, 평균 수강 인원, 최대 수강 인원을 볼 수 있다.")
    lines.append("")

    lines.append("[사진 파일 설명]")
    lines.append("")
    lines.append("1. classroom_period_load_heatmap.png")
    lines.append("- 교시별·교실별 학생 밀집도를 색으로 나타낸 그림이다.")
    lines.append("- 색이 진할수록 해당 교시에 해당 교실에 학생이 많다는 뜻이다.")
    lines.append("- 이 그림은 Wi-Fi 혼잡이 언제, 어느 교실에서 발생할 가능성이 큰지 찾는 데 가장 중요하다.")
    lines.append("")

    lines.append("2. classroom_congestion_score_bar.png")
    lines.append("- 교실별 종합 Wi-Fi 예상 혼잡도 점수를 막대그래프로 나타낸 그림이다.")
    lines.append("- 특정 교시 하나가 아니라 전체 시간표를 종합해서 어느 교실이 전체적으로 혼잡한지 보여준다.")
    lines.append("")

    lines.append("3. classroom_kmeans_scatter.png")
    lines.append("- K-means 알고리즘이 교실을 어떻게 분류했는지 보여주는 산점도이다.")
    lines.append("- x축은 평균 학생 수, y축은 최대 학생 수이다.")
    lines.append("- 위쪽에 있는 교실일수록 특정 시간대에 학생이 많이 몰리는 교실이다.")
    lines.append("")

    lines.append("4. kmeans_elbow_graph.png")
    lines.append("- K-means에서 군집 수 K를 정할 때 참고하는 그래프이다.")
    lines.append("- 그래프가 갑자기 완만해지는 지점이 적절한 K 후보가 된다.")
    lines.append("")

    lines.append("5. global_period_classroom_peak_top20.png")
    lines.append("- 전체 시간표에서 학생 수가 가장 많이 몰린 교시-교실 조합 상위 20개를 보여주는 그래프이다.")
    lines.append("- 이 그림은 실제로 어느 시간에 어느 교실의 Wi-Fi 사용량이 가장 높을지 설명할 때 사용한다.")
    lines.append("")

    lines.append("6. period_top_classrooms 폴더 안의 사진들")
    lines.append("- 각 교시마다 학생 수가 많은 교실 TOP 10을 따로 그린 사진이다.")
    lines.append("- 예를 들어 월_1교시_top_classrooms.png는 월요일 1교시에 어느 교실이 가장 혼잡한지 보여준다.")
    lines.append("- 교시별 원인 분석을 할 때 사용한다.")

    with open(OUT_DIR / "output_file_explanations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 15. 실행
# =========================

def main():
    setup_korean_font()

    timetable_df, period_cols, timetable_path = load_timetable()

    long_df, excluded_df = build_student_location_long(timetable_df, period_cols)
    load = build_classroom_period_load(long_df)
    features = build_classroom_features(load)

    result, cluster_summary, elbow_df = classify_classrooms_kmeans(features)

    course_summary = analyze_course_count_file()

    recommendations = create_recommendations(result)

    plot_period_heatmap(load)
    plot_congestion_bar(result)
    plot_kmeans_scatter(result)
    plot_elbow(elbow_df)
    plot_each_period_top_classrooms(load, top_n=10)
    plot_global_period_classroom_peak(load, top_n=20)

    create_output_file_explanation_txt()

    print("\n완료")
    print(f"결과 폴더: {OUT_DIR}")
    print("")
    print("핵심 CSV:")
    print("- classroom_period_load.csv")
    print("- classroom_congestion_features.csv")
    print("- classroom_kmeans_result.csv")
    print("- classroom_wifi_recommendations.csv")
    print("- period_top_classrooms_summary.csv")
    print("- global_period_classroom_peak_top20.csv")
    print("")
    print("핵심 사진:")
    print("- classroom_period_load_heatmap.png")
    print("- classroom_congestion_score_bar.png")
    print("- classroom_kmeans_scatter.png")
    print("- global_period_classroom_peak_top20.png")
    print("- period_top_classrooms 폴더")
    print("")
    print("설명 TXT:")
    print("- output_file_explanations.txt")


if __name__ == "__main__":
    main()