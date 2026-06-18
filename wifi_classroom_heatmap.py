import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm


CSV_COLUMNS = ["point", "x_m", "y_m", "rssi_dbm", "note"]


def set_korean_font():
    """
    macOS / Windows / Linux에서 한국어 제목이 최대한 깨지지 않게 설정.
    """
    plt.rcParams["axes.unicode_minus"] = False

    candidates = [
        "AppleGothic",       # macOS
        "Malgun Gothic",     # Windows
        "NanumGothic",       # Linux
        "Noto Sans CJK KR",
    ]

    for font in candidates:
        try:
            plt.rcParams["font.family"] = font
            return
        except Exception:
            pass


def init_csv(csv_path: str, width: float, height: float):
    """
    교실 측정용 CSV 템플릿 생성.
    실제 측정 후 rssi_dbm만 수정해서 쓰면 됨.
    """
    points = []

    # 교실 내부를 대략 3 x 3 격자로 나눈 기본 측정점
    xs = np.linspace(width * 0.15, width * 0.85, 3)
    ys = np.linspace(height * 0.15, height * 0.85, 3)

    idx = 1
    for y in ys:
        for x in xs:
            points.append({
                "point": f"P{idx:02d}",
                "x_m": round(float(x), 2),
                "y_m": round(float(y), 2),
                "rssi_dbm": "",
                "note": "직접 측정값 입력"
            })
            idx += 1

    df = pd.DataFrame(points, columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[완료] CSV 템플릿 생성: {csv_path}")
    print("이제 rssi_dbm 칸에 측정한 Wi-Fi 신호 세기 값을 넣으면 됩니다.")
    print("예: -42, -55, -67 처럼 dBm 단위로 입력")


def read_rssi_macos():
    """
    최신 macOS에서는 airport -I가 막히거나 deprecated되어 실패할 수 있음.
    그래서 wdutil info를 먼저 시도하고, 실패하면 airport를 보조로 시도함.
    """
    import re
    import subprocess
    import os

    # 1) 최신 macOS 권장 방식: wdutil
    try:
        result = subprocess.run(
            ["sudo", "wdutil", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )

        output = result.stdout + result.stderr

        # 형식 1: Signal / Noise: -53 dBm / -90 dBm
        match = re.search(r"Signal\s*/\s*Noise:\s*(-?\d+)\s*dBm", output)
        if match:
            return int(match.group(1))

        # 형식 2: RSSI: -53 dBm
        match = re.search(r"RSSI\s*:\s*(-?\d+)\s*dBm", output)
        if match:
            return int(match.group(1))

    except Exception:
        pass

    # 2) 구형 macOS 보조 방식: airport
    airport_paths = [
        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/A/Resources/airport",
    ]

    for airport_path in airport_paths:
        if not os.path.exists(airport_path):
            continue

        try:
            result = subprocess.run(
                [airport_path, "-I"],
                capture_output=True,
                text=True,
                timeout=5
            )

            output = result.stdout + result.stderr

            match = re.search(r"agrCtlRSSI:\s*(-?\d+)", output)
            if match:
                return int(match.group(1))

        except Exception:
            pass

    return None


def append_sample(csv_path: str, point: str, x_m: float, y_m: float, note: str = ""):
    """
    현재 위치의 RSSI를 macOS에서 자동 측정해 CSV에 추가.
    실패하면 사용자에게 수동 입력 요구.
    """
    rssi = read_rssi_macos()

    if rssi is None:
        print("[경고] macOS에서 RSSI 자동 측정에 실패했습니다.")
        print("직접 Wi-Fi RSSI 값을 입력하세요. 예: -55")
        rssi = int(input("RSSI(dBm): ").strip())

    new_row = pd.DataFrame([{
        "point": point,
        "x_m": x_m,
        "y_m": y_m,
        "rssi_dbm": rssi,
        "note": note
    }])

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[완료] 측정값 추가: {point}, x={x_m}, y={y_m}, RSSI={rssi} dBm")


def validate_data(df: pd.DataFrame, width: float, height: float):
    required = {"x_m", "y_m", "rssi_dbm"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {missing}")

    df = df.copy()
    df["x_m"] = pd.to_numeric(df["x_m"], errors="coerce")
    df["y_m"] = pd.to_numeric(df["y_m"], errors="coerce")
    df["rssi_dbm"] = pd.to_numeric(df["rssi_dbm"], errors="coerce")

    df = df.dropna(subset=["x_m", "y_m", "rssi_dbm"])

    if len(df) < 3:
        raise ValueError("히트맵을 만들려면 최소 3개 이상의 측정점이 필요합니다.")

    out_of_room = df[
        (df["x_m"] < 0) |
        (df["x_m"] > width) |
        (df["y_m"] < 0) |
        (df["y_m"] > height)
    ]

    if len(out_of_room) > 0:
        print("[경고] 교실 범위를 벗어난 측정점이 있습니다.")
        print(out_of_room)

    return df


def idw_interpolation(points, values, grid_x, grid_y, power=2.0, eps=1e-9):
    """
    IDW(Inverse Distance Weighting) 보간법.
    가까운 측정점의 영향을 더 크게 반영함.

    추정값 = sum(측정값 / 거리^p) / sum(1 / 거리^p)
    """
    xi = grid_x.ravel()
    yi = grid_y.ravel()

    estimated = np.zeros_like(xi, dtype=float)
    weight_sum = np.zeros_like(xi, dtype=float)

    for (px, py), val in zip(points, values):
        dist = np.sqrt((xi - px) ** 2 + (yi - py) ** 2)
        weights = 1.0 / ((dist + eps) ** power)

        estimated += weights * val
        weight_sum += weights

    estimated = estimated / weight_sum
    return estimated.reshape(grid_x.shape)


def classify_rssi(rssi):
    """
    RSSI 해석용 간단 기준.
    """
    if rssi >= -50:
        return "매우 강함"
    elif rssi >= -60:
        return "양호"
    elif rssi >= -70:
        return "보통"
    elif rssi >= -80:
        return "약함"
    else:
        return "매우 약함"


def plot_heatmap(
    csv_path: str,
    width: float,
    height: float,
    output: str,
    title: str,
    floorplan: str | None = None,
    resolution: int = 250,
    power: float = 2.0,
    extreme: bool = True,
):
    set_korean_font()

    df = pd.read_csv(csv_path)
    df = validate_data(df, width, height)

    points = df[["x_m", "y_m"]].to_numpy(dtype=float)
    values = df["rssi_dbm"].to_numpy(dtype=float)

    grid_x, grid_y = np.meshgrid(
        np.linspace(0, width, resolution),
        np.linspace(0, height, resolution)
    )

    grid_rssi = idw_interpolation(
        points=points,
        values=values,
        grid_x=grid_x,
        grid_y=grid_y,
        power=power
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    # 도면 배경
    if floorplan:
        if not os.path.exists(floorplan):
            raise FileNotFoundError(f"도면 이미지가 없습니다: {floorplan}")

        img = plt.imread(floorplan)
        ax.imshow(
            img,
            extent=[0, width, 0, height],
            origin="upper",
            alpha=0.25,
            aspect="auto"
        )

    if extreme:
        # 측정값 범위가 좁아도 색 차이가 크게 보이도록 분위수 기반 색상 범위 사용
        vmin = np.percentile(grid_rssi, 5)
        vmax = np.percentile(grid_rssi, 95)

        # 값 범위가 너무 좁을 때 최소 대비 확보
        if vmax - vmin < 5:
            center = (vmax + vmin) / 2
            vmin = center - 3
            vmax = center + 3

        # 약한 신호 영역을 더 강하게 드러내기 위한 비선형 색상 정규화
        norm = PowerNorm(gamma=0.65, vmin=vmin, vmax=vmax)

        heatmap = ax.imshow(
            grid_rssi,
            extent=[0, width, 0, height],
            origin="lower",
            aspect="auto",
            cmap="turbo",
            norm=norm,
            alpha=0.92
        )

        contour_levels = np.linspace(vmin, vmax, 8)
        contours = ax.contour(
            grid_x,
            grid_y,
            grid_rssi,
            levels=contour_levels,
            colors="black",
            linewidths=0.45,
            alpha=0.45
        )

        ax.clabel(
            contours,
            inline=True,
            fontsize=7,
            fmt="%.1f"
        )

        color_note = f"색상 범위: 하위 5%~상위 95% 강조 ({vmin:.1f} ~ {vmax:.1f} dBm)"

    else:
        # 덜 과장된 일반 표현
        heatmap = ax.imshow(
            grid_rssi,
            extent=[0, width, 0, height],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            alpha=0.85
        )

        color_note = "색상 범위: 전체 측정값 기준"

    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label("Wi-Fi 신호 세기 RSSI (dBm)")

    # 측정 지점 표시
    ax.scatter(
        df["x_m"],
        df["y_m"],
        s=85,
        c="white",
        edgecolors="black",
        linewidths=1.2,
        label="측정 지점",
        zorder=5
    )

    for _, row in df.iterrows():
        point_name = str(row["point"]) if "point" in df.columns else ""
        label = f"{point_name}\n{int(row['rssi_dbm'])} dBm"

        ax.text(
            row["x_m"],
            row["y_m"] + 0.08,
            label,
            fontsize=8,
            ha="center",
            va="bottom",
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                alpha=0.75,
                boxstyle="round,pad=0.2"
            ),
            zorder=6
        )

    ax.set_title(title + "\n" + color_note)
    ax.set_xlabel("교실 가로 위치 x (m)")
    ax.set_ylabel("교실 세로 위치 y (m)")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()

    grid_output = Path(output).with_suffix(".grid.csv")
    grid_df = pd.DataFrame({
        "x_m": grid_x.ravel(),
        "y_m": grid_y.ravel(),
        "estimated_rssi_dbm": grid_rssi.ravel()
    })
    grid_df.to_csv(grid_output, index=False, encoding="utf-8-sig")

    print(f"[완료] 극단 강조 히트맵 저장: {output}")
    print(f"[완료] 보간 데이터 저장: {grid_output}")

    print("\n[측정값 요약]")
    print(f"측정점 개수: {len(df)}개")
    print(f"최고 신호: {values.max():.1f} dBm ({classify_rssi(values.max())})")
    print(f"최저 신호: {values.min():.1f} dBm ({classify_rssi(values.min())})")
    print(f"평균 신호: {values.mean():.1f} dBm ({classify_rssi(values.mean())})")
    print(f"신호 범위: {values.max() - values.min():.1f} dB")


def main():
    parser = argparse.ArgumentParser(
        description="한 교실 Wi-Fi RSSI 히트맵 생성 스크립트"
    )

    parser.add_argument("--csv", default="classroom_wifi_points.csv")
    parser.add_argument("--width", type=float, default=8.0, help="교실 가로 길이(m)")
    parser.add_argument("--height", type=float, default=7.0, help="교실 세로 길이(m)")

    parser.add_argument("--init", action="store_true", help="측정용 CSV 템플릿 생성")
    parser.add_argument("--plot", action="store_true", help="히트맵 생성")

    parser.add_argument("--sample", action="store_true", help="현재 RSSI를 측정해 CSV에 추가")
    parser.add_argument("--point", default="P_new", help="측정 지점 이름")
    parser.add_argument("--x", type=float, help="측정 위치 x좌표(m)")
    parser.add_argument("--y", type=float, help="측정 위치 y좌표(m)")
    parser.add_argument("--note", default="", help="측정 지점 설명")

    parser.add_argument("--floorplan", default=None, help="교실 도면 이미지 파일 경로")
    parser.add_argument("--output", default="wifi_heatmap.png")
    parser.add_argument("--title", default="교실 Wi-Fi 신호 세기 히트맵")
    parser.add_argument("--resolution", type=int, default=250)
    parser.add_argument("--power", type=float, default=2.0, help="IDW 거리 가중치 지수")
    parser.add_argument("--normal", action="store_true", help="극단 강조 없이 일반 히트맵 생성")

    args = parser.parse_args()

    if args.init:
        init_csv(args.csv, args.width, args.height)
        return

    if args.sample:
        if args.x is None or args.y is None:
            raise ValueError("--sample 사용 시 --x, --y 좌표가 필요합니다.")

        append_sample(
            csv_path=args.csv,
            point=args.point,
            x_m=args.x,
            y_m=args.y,
            note=args.note
        )
        return

    if args.plot:
        plot_heatmap(
    csv_path=args.csv,
    width=args.width,
    height=args.height,
    output=args.output,
    title=args.title,
    floorplan=args.floorplan,
    resolution=args.resolution,
    power=args.power,
    extreme=not args.normal
)
        return

    print("사용 예시:")
    print("1) CSV 템플릿 생성")
    print("python wifi_classroom_heatmap.py --init --width 8 --height 7")
    print()
    print("2) 현재 위치 RSSI 자동 측정 후 CSV 추가")
    print("python wifi_classroom_heatmap.py --sample --point P01 --x 1.0 --y 1.0")
    print()
    print("3) 히트맵 생성")
    print("python wifi_classroom_heatmap.py --plot --width 8 --height 7")


if __name__ == "__main__":
    main()