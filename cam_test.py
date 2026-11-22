#!/usr/bin/env python3
"""C270n の動作確認用：映像を表示し、スナップショットを保存する。

使い方:
    python cam_test.py          # カメラ番号を自動で探す（内蔵カメラ以外を優先）
    python cam_test.py 1        # カメラ番号を指定

キー操作（映像のウィンドウを選択した状態で）:
    r : 解像度を切り替え（1280x720 ⇔ 640x480）
    e : 自動露出 ON/OFF（OFFにすると明るさが固定され、fpsも安定しやすい）
    + / - : 露出を明るく / 暗く（自動露出OFFのとき）
    s : スナップショットを保存（snap_YYYYmmdd_HHMMSS.png）
    g : 中央の十字と目盛り線の表示を切り替え
    q : 終了
"""
import sys
import time
from datetime import datetime

import cv2

SIZES = [(1280, 720), (640, 480)]


def fourcc_str(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))


def configure(cap, size):
    w, h = size
    # MJPG（圧縮）で受け取る。非圧縮(YUY2)だとUSB2の帯域で720pは約7fpsに落ちる
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.read()
    print(f"解像度 {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f"  形式 {fourcc_str(cap)}  設定fps {cap.get(cv2.CAP_PROP_FPS):.0f}")


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def find_camera():
    for idx in [1, 2, 3, 0]:  # 0は内蔵カメラのことが多いので後回し
        cap = open_camera(idx)
        if cap is not None:
            print(f"camera index {idx} を使います")
            return cap, idx
    return None, None


if len(sys.argv) > 1:
    idx = int(sys.argv[1])
    cap = open_camera(idx)
else:
    cap, idx = find_camera()

if cap is None:
    sys.exit("カメラが開けません。USBの接続と、他のアプリ（Zoom等）が使っていないか確認してください。")

size_i = 0
configure(cap, SIZES[size_i])
auto_exp = True
exposure = -6
print("r:解像度  e:自動露出  +/-:露出  s:保存  g:ガイド  q:終了")

show_guide = True
t_prev, fps = time.time(), 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("フレームを取得できませんでした")
        break
    h, w = frame.shape[:2]

    now = time.time()
    fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
    t_prev = now

    view = frame.copy()
    if show_guide:
        cx, cy = w // 2, h // 2
        cv2.line(view, (cx, 0), (cx, h), (0, 255, 0), 1)
        cv2.line(view, (0, cy), (w, cy), (0, 255, 0), 1)
        for x in range(0, w, 100):
            cv2.line(view, (x, cy - 8), (x, cy + 8), (0, 255, 0), 1)
    exp_txt = "AE" if auto_exp else f"exp {exposure}"
    cv2.putText(view, f"cam {idx}  {w}x{h}  {fps:4.1f} fps  {exp_txt}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("cam_test", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("g"):
        show_guide = not show_guide
    elif key == ord("r"):
        size_i = (size_i + 1) % len(SIZES)
        configure(cap, SIZES[size_i])
    elif key == ord("e"):
        auto_exp = not auto_exp
        # DirectShow: 0.75=自動, 0.25=手動
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_exp else 0.25)
        if not auto_exp:
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        print("自動露出", "ON" if auto_exp else f"OFF (exposure={exposure})")
    elif key in (ord("+"), ord(";")) and not auto_exp:
        exposure = min(exposure + 1, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    elif key == ord("-") and not auto_exp:
        exposure = max(exposure - 1, -13)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    elif key == ord("s"):
        name = datetime.now().strftime("snap_%Y%m%d_%H%M%S.png")
        cv2.imwrite(name, frame)
        print(f"保存: {name}")

cap.release()
cv2.destroyAllWindows()
