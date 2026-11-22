#!/usr/bin/env python3
"""カメラで物体の幅を測る（背景差分方式）。

考え方:
    1. 何も置かない状態（グリッパーは開いた位置）を「背景」として記録する
    2. 物体を置くと、背景と違う部分＝物体として切り出せる
    3. 物体を囲む最小の長方形を求め、短い辺＝幅[mm]、長い辺＝長さ[mm] とする
    背景の色に関係なく使えるので、白い紙の上の白っぽい物でも、ある程度は切り出せる。

使い方:
    python vision_measure.py        # カメラ番号は自動（内蔵以外を優先）
    python vision_measure.py 1      # カメラ番号を指定

最初の1回だけやる設定（vision_config.json に保存され、次回から自動で読み込む）:
    a : 測る範囲(ROI)をマウスで囲んで Enter（指の間と物体を置く場所を囲む）
    k : 長さの校正。方眼の上で、離れた2点をクリック → 映像のウィンドウで実際の距離[mm]を
        数字キーで入力して Enter（Backspaceで1文字消す、Escで取り消し）

毎回の操作:
    b : 背景を記録（物体を置く前に。グリッパーは開いた位置で）
    m : 測定（15フレームの中央値をPowerShellに表示）
    q : 終了
    スライダー「thresh」: 背景との違いをどれだけで物体とみなすか（影まで拾うなら上げる）
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CONFIG = Path(__file__).with_name("vision_config.json")
WIN = "vision_measure"


# ---------------- カメラ ----------------
def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def find_camera(arg):
    order = [int(arg)] if arg is not None else [1, 2, 3, 0]
    for idx in order:
        cap = open_camera(idx)
        if cap is not None:
            print(f"camera index {idx} を使います")
            return cap
    sys.exit("カメラが開けません。USBの接続と、他のアプリが使っていないか確認してください。")


# ---------------- 設定 ----------------
def load_config():
    if CONFIG.exists():
        return json.loads(CONFIG.read_text())
    return {"roi": None, "px_per_mm": None}


def save_config(cfg):
    CONFIG.write_text(json.dumps(cfg, indent=2))
    print(f"保存: {CONFIG.name}  {cfg}")


# ---------------- 測定の本体 ----------------
class Measurer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bg = None
        self.ignore = None  # 無視する画素（指の縁など、毎回少し動く所）。ROIと同じ大きさのbool配列

    def crop(self, frame):
        r = self.cfg["roi"]
        if r is None:
            return frame, (0, 0)
        x, y, w, h = r
        return frame[y:y + h, x:x + w], (x, y)

    def set_background(self, frames):
        crops = [self.crop(f)[0].astype(np.float32) for f in frames]
        self.bg = np.median(crops, axis=0).astype(np.uint8)
        self.ignore = None
        print("背景を記録しました")

    def set_ignore_from_frames(self, frames, thresh=25, grow=7):
        """指を少し動かした前後のフレームから、変化した画素（指の縁）を「無視する所」として記録"""
        crops = [cv2.cvtColor(self.crop(f)[0], cv2.COLOR_BGR2GRAY).astype(np.int16) for f in frames]
        moved = np.zeros(crops[0].shape, bool)
        for c in crops[1:]:
            moved |= np.abs(c - crops[0]) > thresh
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
        self.ignore = cv2.dilate(moved.astype(np.uint8), k) > 0
        print(f"指の縁を無視する範囲を記録しました（{int(self.ignore.sum())}画素）")

    def measure(self, frame, thresh):
        """戻り値: (結果dict または None, 二値マスク, ROIの左上座標)"""
        roi, off = self.crop(frame)
        if self.bg is None or self.bg.shape != roi.shape:
            return None, None, off
        # 明るさ(L)と色(a,b)の違いをまとめて見る。白っぽい物は色より明るさや模様で差が出る
        lab_now = cv2.cvtColor(cv2.GaussianBlur(roi, (3, 3), 0), cv2.COLOR_BGR2LAB)
        lab_bg = cv2.cvtColor(cv2.GaussianBlur(self.bg, (3, 3), 0), cv2.COLOR_BGR2LAB)
        now16, bg16 = lab_now.astype(np.int16), lab_bg.astype(np.int16)
        dL = now16[:, :, 0] - bg16[:, :, 0]                          # 明るさの変化（+で明るくなった）
        dC = np.abs(now16[:, :, 1:] - bg16[:, :, 1:]).max(axis=2)    # 色の変化
        if self.cfg.get("ignore_dark", True):
            # 影対策：「暗くなっただけで色は変わらない」部分は影とみなして無視する
            # （暗い机の上の明るい物体向け。白い紙の上で暗い物を測るときは ignore_dark を false に）
            fg = (dL > thresh) | (dC > thresh)
        else:
            fg = (np.abs(dL) > thresh) | (dC > thresh)
        if self.ignore is not None and self.ignore.shape == fg.shape:
            fg &= ~self.ignore
        mask = fg.astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)               # 細かいノイズを消す
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)  # 物体の中の穴を埋める

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask, off
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 150:  # 小さすぎるものはノイズ扱い
            return None, mask, off

        rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
        (cx, cy), (rw, rh), ang = rect
        short_px, long_px = sorted([rw, rh])
        ppm = self.cfg["px_per_mm"]
        res = {
            "width_px": short_px,
            "length_px": long_px,
            "width_mm": short_px / ppm if ppm else None,
            "length_mm": long_px / ppm if ppm else None,
            "center": (cx + off[0], cy + off[1]),
            "box": cv2.boxPoints(rect) + np.array(off, dtype=np.float32),
            "area_px": cv2.contourArea(c),
        }
        return res, mask, off


# ---------------- 校正（2点クリック） ----------------
clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and param["calib"]:
        clicks.append((x, y))
        print(f"クリック {len(clicks)}: ({x}, {y})")


def main():
    cap = find_camera(sys.argv[1] if len(sys.argv) > 1 else None)
    cfg = load_config()
    meas = Measurer(cfg)
    state = {"calib": False, "typing": None}  # typing: 距離入力中の文字列

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse, state)
    cv2.createTrackbar("thresh", WIN, 25, 100, lambda v: None)
    print(__doc__.split("最初の1回だけ")[1])

    t_prev, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now

        thresh = max(cv2.getTrackbarPos("thresh", WIN), 1)
        res, mask, off = meas.measure(frame, thresh)
        view = frame.copy()

        if cfg["roi"]:
            x, y, w, h = cfg["roi"]
            cv2.rectangle(view, (x, y), (x + w, y + h), (255, 200, 0), 1)
        if mask is not None:  # 物体と判定した部分を赤く重ねる
            roi_view = view[off[1]:off[1] + mask.shape[0], off[0]:off[0] + mask.shape[1]]
            roi_view[mask > 0] = (0.5 * roi_view[mask > 0] + [0, 0, 127]).astype(np.uint8)
        if res:
            cv2.drawContours(view, [res["box"].astype(int)], 0, (0, 255, 0), 2)
            txt = (f"W {res['width_mm']:.1f}mm  L {res['length_mm']:.1f}mm" if res["width_mm"]
                   else f"W {res['width_px']:.0f}px  L {res['length_px']:.0f}px (k: not calibrated)")
            cv2.putText(view, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if state["typing"] is not None:
            status = f"distance mm: {state['typing']}_  (Enter)"
        elif state["calib"]:
            status = "calib: click 2 points"
        else:
            status = "b: no background" if meas.bg is None else ""
        cv2.putText(view, f"{fps:4.1f} fps  {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for p in clicks:
            cv2.circle(view, p, 4, (0, 0, 255), -1)
        cv2.imshow(WIN, view)

        key = cv2.waitKey(1) & 0xFF

        # 距離の入力中は、キーを数字入力として扱う
        if state["typing"] is not None:
            if key in (13, 10):  # Enter
                try:
                    mm = float(state["typing"])
                    (x1, y1), (x2, y2) = clicks[:2]
                    dist_px = float(np.hypot(x2 - x1, y2 - y1))
                    cfg["px_per_mm"] = dist_px / mm
                    print(f"2点の距離 {dist_px:.1f}px = {mm}mm → {cfg['px_per_mm']:.3f} px/mm")
                    save_config(cfg)
                except (ValueError, ZeroDivisionError):
                    print("数字が読めなかったので、校正を取り消しました")
                state["typing"] = None
                clicks.clear()
            elif key == 27:  # Esc
                state["typing"] = None
                clicks.clear()
                print("校正を取り消しました")
            elif key == 8:  # Backspace
                state["typing"] = state["typing"][:-1]
            elif key != 255 and chr(key) in "0123456789.":
                state["typing"] += chr(key)
            continue

        if key == ord("q"):
            break

        elif key == ord("a"):
            r = cv2.selectROI(WIN, frame, showCrosshair=True)
            cv2.setMouseCallback(WIN, on_mouse, state)  # selectROIに上書きされたクリック処理を戻す
            if r[2] > 0 and r[3] > 0:
                cfg["roi"] = [int(v) for v in r]
                meas.bg = None
                save_config(cfg)
                print("範囲を変えたので、もう一度 b で背景を記録してください")

        elif key == ord("k"):
            clicks.clear()
            state["calib"] = True
            print("方眼の上で、離れた2点をクリックしてください（例：5cm離れた線の交点）")

        elif key == ord("b"):
            frames = [cap.read()[1] for _ in range(10)]
            meas.set_background(frames)

        elif key == ord("m"):
            ws, ls = [], []
            for _ in range(15):
                r, _, _ = meas.measure(cap.read()[1], thresh)
                if r:
                    ws.append(r["width_px"])
                    ls.append(r["length_px"])
            if not ws:
                print("物体が見つかりません（b で背景を記録したか、thresh が高すぎないか確認）")
            else:
                w_px, l_px = float(np.median(ws)), float(np.median(ls))
                ppm = cfg["px_per_mm"]
                if ppm:
                    print(f"幅 {w_px / ppm:.1f} mm   長さ {l_px / ppm:.1f} mm   ({len(ws)}/15フレーム)")
                else:
                    print(f"幅 {w_px:.0f} px   長さ {l_px:.0f} px   （k で校正するとmmになります）")

        # 2点そろったら、距離の入力モードへ
        if state["calib"] and len(clicks) >= 2:
            state["calib"] = False
            state["typing"] = ""
            print("映像のウィンドウで、2点の実際の距離[mm]を数字キーで入力して Enter")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
