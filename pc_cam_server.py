"""
pc_cam_server.py
170°魚眼カメラ → VR ゴーグル用左右映像変換ストリーミング
-------------------------------------------------------------
● 等距離（f-θ）モデルで逆射影し，中心部の拡大を防止
● 事前計算したマップを使い，毎フレーム軽量にリマップ
● `eye_offset_deg` ≈ ±3–6° で擬似的に両眼視差を付与
"""

import cv2
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

# ==== パラメータ ====
FOV_DEG = 170  # カメラ水平視野
EYE_SEP_DEG = 5  # 眼間のヨー回転(片目分)
CAM_ID = 1  # VideoCapture のデバイス番号
OUT_SIZE = 720  # 片目画像の一辺（正方形で出力）
# =====================


def build_maps(
    out_len: int,
    in_h: int,
    in_w: int,
    fov_deg: float = 170.0,
    eye_offset_deg: float = 0.0,
):
    """
    出力→入力への逆写像マップを生成（等距離投影）。
    一度だけ呼び出し，返ってきた map_x/map_y を remap に渡す。
    """
    fov = np.radians(fov_deg)
    yaw = np.radians(eye_offset_deg)

    # 出力座標を -1〜1 に正規化
    uv = np.linspace(-1.0, 1.0, out_len, dtype=np.float32)
    u, v = np.meshgrid(uv, uv)  # u:右+  v:下+

    r = np.sqrt(u**2 + v**2)  # 正規化半径
    theta = r * (fov / 2)  # 等距離モデル: r ∝ θ

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    denom = np.where(r == 0, 1, r)  # 0 除算回避
    X = sin_t * (u / denom)
    Y = sin_t * (v / denom)
    Z = cos_t

    # ヨー回転 (左右目で±)
    Xr = X * np.cos(yaw) + Z * np.sin(yaw)
    Yr = Y
    Zr = -X * np.sin(yaw) + Z * np.cos(yaw)

    # 等距離: r_f = f * θ'
    theta_r = np.arccos(Zr)
    f = (in_w / 2) / (fov / 2)
    rf = f * theta_r

    norm = np.sqrt(Xr**2 + Yr**2)
    norm = np.where(norm == 0, 1, norm)
    map_x = (in_w / 2 + rf * Xr / norm).astype(np.float32)
    map_y = (in_h / 2 + rf * Yr / norm).astype(np.float32)
    return map_x, map_y


def create_eye_warpers(cam_h: int, cam_w: int):
    """左右目用のマップを生成し，高速ワーパー関数を返す"""
    lmx, lmy = build_maps(OUT_SIZE, cam_h, cam_w, FOV_DEG, -EYE_SEP_DEG)
    rmx, rmy = build_maps(OUT_SIZE, cam_h, cam_w, FOV_DEG, +EYE_SEP_DEG)

    def warp_left(frame):  # 事前計算マップでリマップ
        return cv2.remap(
            frame, lmx, lmy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

    def warp_right(frame):
        return cv2.remap(
            frame, rmx, rmy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

    return warp_left, warp_right


def gen_frames():
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        raise RuntimeError("カメラが開けません")

    # カメラサイズ取得 → マップ生成
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    warp_left, warp_right = create_eye_warpers(cam_h, cam_w)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        left_eye = warp_left(frame)
        right_eye = warp_right(frame)
        combined = np.hstack((left_eye, right_eye))  # L|R

        # Motion JPEG でストリーミング
        _, buf = cv2.imencode(".jpg", combined)
        yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/")
def stream():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
