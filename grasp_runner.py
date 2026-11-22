#!/usr/bin/env python3
"""カメラ＋グリッパーで把持を1回ずつ実行し、結果をCSVに記録する。

3つの条件を切り替えて、同じ物体で比較実験ができる:
    1 固定制御     : カメラも触覚も使わず、決まった位置(FIXED_US)まで閉じる
    2 視覚のみ     : カメラで測った幅に合わせて閉じる（触覚は使わない）
    3 視覚＋触覚   : カメラの幅＋余裕まで素早く閉じ → そこからFSRと位置を見ながら
                     ゆっくり閉じて止める(Arduinoの g)。空振り(LIMIT)なら1回やり直す

使い方:
    （ArduinoのシリアルモニタとPowerShellのlogger.pyは閉じておく）
    python grasp_runner.py --obj sponge --tag v1        （COM番号は自動で探す）
    オプション: --cam 1（カメラ番号）  --log logs（保存フォルダ）
                --tag v1（設定の版。スケッチや設定を変えたら v2 などに変える）

キー操作（映像のウィンドウで）:
    c       : グリッパーの自動校正（何も挟まずに。電源を入れたら最初に1回）
    o       : グリッパーを開く
    b       : 背景を記録（何も置かない状態で。指が少し動いて、指の縁を無視する範囲も記録する）
    1 2 3   : 条件を切り替え（固定 / 視覚のみ / 視覚＋触覚）
    スペース: 把持を1回実行（物体を指の間に置いてから）
    y / n   : 直前の把持の結果を「成功 / 失敗」として記録 → 自動で開く
    x       : 直前の把持を記録せずに捨てる → 自動で開く
    q       : 終了

記録されるファイル:
    logs/trials.csv          : 1回の把持ごとに1行（比較実験の集計用）
    logs/raw_<日時>_<条件>.csv : 把持中のArduinoの全データ（グラフ用）

測る範囲(a)と長さの校正(k)は vision_measure.py で済ませておく（vision_config.json を共有）。
"""
import argparse
import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial
import serial.tools.list_ports

from vision_measure import Measurer, load_config, open_camera

# ===== 実験の設定（必要に応じて書き換える）=====
FIXED_US = 2120          # 条件1：固定制御で閉じる位置（全閉）
VISION_SQUEEZE_MM = 2.0  # 条件2：カメラの幅からこれだけ小さい位置まで閉じる
PRECLOSE_MARGIN_MM = 3.0 # 条件3：カメラの幅＋これだけ の位置まで素早く閉じる
HOLD_TIME_S = 1.0        # 閉じ終わってから結果を読むまで待つ時間
GRASP_TIMEOUT_S = 8.0    # 条件3で、g の結果を待つ最大時間
MAX_RETRY = 1            # 条件3で、空振り(LIMIT)のときにやり直す回数

# gripper_test.ino の WID_US / WID_MM と同じ表（指令パルス幅 → 指先の隙間）
WID_US = [560, 1000, 1500, 2000, 2150]
WID_MM = [27.7, 19.0, 11.4, 4.6, 3.2]
PULSE_MIN, PULSE_MAX = 560, 2150
# ===============================================

COND_NAMES = {1: "fixed", 2: "vision", 3: "vision_touch"}
COND_LABEL = {1: "1 fixed", 2: "2 vision only", 3: "3 vision + touch"}


def mm_to_us(mm):
    """指先の隙間[mm] → 指令パルス幅[us]（表を逆に補間）"""
    us = np.interp(mm, WID_MM[::-1], WID_US[::-1])
    return int(np.clip(us, PULSE_MIN, PULSE_MAX))


# ---------------- Arduinoとの通信 ----------------
def find_arduino_port():
    """接続中のポートから Arduino を探す。見つからなければ一覧を表示して None"""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if p.vid == 0x2341 or "arduino" in text:
            return p.device
    print("Arduinoが見つかりません。今あるポート:")
    for p in ports:
        print(f"   {p.device}: {p.description}")
    if not ports:
        print("   （なし）USBケーブルを抜き挿ししてください")
    return None


class Gripper:
    FIELDS = ["ms", "us", "fsrL", "fsrR", "fb", "diff", "wmm", "state"]

    def __init__(self, port):
        self.ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)  # 接続時のリセット待ち
        self.send("r")  # データの出力を再開させる（前回 p で止めている場合に備えて）
        self.latest = None
        self.latest_t = 0.0     # 最後にデータを受け取った時刻
        self.lock = threading.Lock()
        self.raw_file = None
        self.running = True
        self.unparsed = []  # 読めなかった行（診断用）
        threading.Thread(target=self._reader, daemon=True).start()
        # 3秒以内にデータが来なければ、原因の手がかりを表示
        t0 = time.time()
        while time.time() - t0 < 3 and self.get() is None:
            time.sleep(0.1)
        if self.get() is None:
            print(f"!! {port} からArduinoのデータを読めません")
            if self.unparsed:
                print("   受信した行の例（列の数が合っていない＝スケッチの版が違う可能性）:")
                for l in self.unparsed[:3]:
                    print("   ", l)
                print(f"   期待している列: {','.join(self.FIELDS)}")
            else:
                print("   何も受信していません。Arduinoが固まっている可能性があります")
                print("   → q で終了 → USBケーブルを抜いて5秒待って挿し直す → 起動し直す")

    def _reader(self):
        while self.running:
            line = self.ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            if line.startswith("#"):
                print(line)
                continue
            parts = line.split(",")
            if parts[0] == "ms":
                continue
            if len(parts) != len(self.FIELDS):
                if len(self.unparsed) < 5:
                    self.unparsed.append(line)
                continue
            try:
                d = {k: (v if k == "state" else float(v)) for k, v in zip(self.FIELDS, parts)}
            except ValueError:
                continue
            with self.lock:
                self.latest = d
                self.latest_t = time.time()
                if self.raw_file:
                    self.raw_file.write(f"{time.time():.3f},{line}\n")

    def send(self, cmd):
        self.ser.write((cmd + "\n").encode())

    def get(self, max_age=0.5):
        """最新のデータ。max_age秒より古い（データが止まっている）ときは None"""
        with self.lock:
            if not self.latest or time.time() - self.latest_t > max_age:
                return None
            return dict(self.latest)

    def ensure_stream(self):
        """データが止まっていたら r を送って再開を試みる。戻り値: 流れているか"""
        if self.get() is not None:
            return True
        for _ in range(3):
            self.send("r")
            time.sleep(0.5)
            if self.get() is not None:
                print("Arduinoのデータ出力を再開しました")
                return True
        return False

    def wait_state(self, states, timeout):
        """state が states のどれかになるまで待つ。戻り値: 最後のデータ"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            d = self.get()
            if d and d["state"] in states:
                return d
            time.sleep(0.02)
        return self.get()

    def start_raw(self, path):
        f = open(path, "w")
        f.write("pc_time," + ",".join(self.FIELDS) + "\n")
        with self.lock:
            self.raw_file = f

    def stop_raw(self):
        with self.lock:
            f, self.raw_file = self.raw_file, None
        if f:
            f.close()

    def close(self):
        # 終了前にArduinoの出力を止める。流しっぱなしでポートを閉じると、
        # 次に開いたときにデータが来なくなることがあるため
        self.send("p")
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        self.running = False
        time.sleep(0.1)
        self.ser.close()


# ---------------- 背景の記録 ----------------
def capture_background(grip, cap, meas):
    """開いた状態を背景として記録し、指を少し動かして「指の縁」を無視する範囲を作る"""
    grip.send("o")
    time.sleep(0.6)
    for _ in range(5):
        cap.read()  # 古いフレームを捨てる
    meas.set_background([cap.read()[1] for _ in range(10)])
    shots = [cap.read()[1]]
    for us in (PULSE_MIN + 60, PULSE_MIN + 30):   # 指を少し閉じた位置
        grip.send(f"u{us}")
        time.sleep(0.5)
        cap.read(); cap.read()
        shots.append(cap.read()[1])
    grip.send("o")
    time.sleep(0.5)
    cap.read(); cap.read()
    shots.append(cap.read()[1])  # 開き直した位置（ガタでずれる分）
    meas.set_ignore_from_frames(shots)


# ---------------- 1回の把持 ----------------
def measure_width(cap, meas, thresh, n=15):
    ws, ls = [], []
    for _ in range(n):
        r, _, _ = meas.measure(cap.read()[1], thresh)
        if r and r["width_mm"]:
            ws.append(r["width_mm"])
            ls.append(r["length_mm"])
    if len(ws) < n // 2:
        return None, None
    return float(np.median(ws)), float(np.median(ls))


def run_trial(cond, grip, cap, meas, thresh):
    """戻り値: 記録用のdict（失敗時は None とメッセージ）"""
    if not grip.ensure_stream():
        return None, ("Arduinoからのデータが止まっています。q で終了 → USBを抜き挿し → 起動し直してください"
                      "（この状態で把持すると、古いデータが記録されてしまいます）")
    res = {"condition": COND_NAMES[cond], "cam_width_mm": "", "cam_length_mm": "",
           "target_us": "", "preclose_us": "", "retries": 0}

    if cond in (2, 3):
        w, l = measure_width(cap, meas, thresh)
        if w is None:
            return None, "物体が見つかりません（b で背景を記録したか、thresh を確認）"
        res["cam_width_mm"], res["cam_length_mm"] = round(w, 1), round(l, 1)

    t0 = time.time()
    if cond == 1:
        res["target_us"] = FIXED_US
        grip.send(f"u{FIXED_US}")
        time.sleep(HOLD_TIME_S)
        d = grip.get()

    elif cond == 2:
        target = mm_to_us(res["cam_width_mm"] - VISION_SQUEEZE_MM)
        res["target_us"] = target
        grip.send(f"u{target}")
        time.sleep(HOLD_TIME_S)
        d = grip.get()

    else:
        pre = mm_to_us(res["cam_width_mm"] + PRECLOSE_MARGIN_MM)
        res["preclose_us"] = pre
        for attempt in range(MAX_RETRY + 1):
            res["retries"] = attempt
            grip.send(f"u{pre}")
            time.sleep(0.4)
            grip.send("g")
            time.sleep(0.1)
            d = grip.wait_state({"HOLDING", "STALL", "LIMIT"}, GRASP_TIMEOUT_S)
            if d and d["state"] != "LIMIT":
                break
            # 空振り：開いて、少し広めの位置からやり直す（適応）
            print("  空振り(LIMIT) → 開いてやり直し")
            grip.send("o")
            time.sleep(0.6)
            pre = max(PULSE_MIN, pre - 150)
        time.sleep(0.3)
        d = grip.get()

    res["grasp_time_s"] = round(time.time() - t0, 2)
    if d is None:
        return None, "把持の途中でArduinoからのデータが止まりました。この試行は記録しません"
    res.update({
        "final_us": int(d["us"]), "fb_mV": int(d["fb"]), "diff": int(d["diff"]),
        "wmm": d["wmm"], "fsrL": int(d["fsrL"]), "fsrR": int(d["fsrR"]),
        "fsr_max": int(max(d["fsrL"], d["fsrR"])), "state": d["state"],
    })
    if res["cam_width_mm"] != "":
        res["cam_minus_finger_mm"] = round(res["cam_width_mm"] - d["wmm"], 1)
    return res, None


# ---------------- メイン ----------------
CSV_FIELDS = ["time", "tag", "object", "condition", "trial_no", "result",
              "cam_width_mm", "cam_length_mm", "target_us", "preclose_us", "retries",
              "final_us", "fb_mV", "diff", "wmm", "cam_minus_finger_mm",
              "fsrL", "fsrR", "fsr_max", "state", "grasp_time_s", "raw_file"]


def migrate_csv(path):
    """古い版の trials.csv（列が違う）を、今の列に合わせて書き直す"""
    if not path.exists():
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == CSV_FIELDS:
            return
        rows = list(reader)
    for r in rows:
        r.setdefault("tag", "v1")
        if not r.get("tag"):
            r["tag"] = "v1"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{path.name} を新しい列の形式に書き直しました（{len(rows)}行）")


def append_csv(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="auto", help="COM番号。省略すると自動で探す")
    ap.add_argument("--cam", type=int, default=None)
    ap.add_argument("--obj", default="object")
    ap.add_argument("--log", default="logs")
    ap.add_argument("--tag", default="v1", help="設定の版の名前（改良の前後を区別する。例: v1, v2_fast）")
    args = ap.parse_args()

    logdir = Path(args.log)
    logdir.mkdir(exist_ok=True)
    trials_csv = logdir / "trials.csv"
    migrate_csv(trials_csv)

    cfg = load_config()
    if not cfg.get("px_per_mm"):
        print("注意: 長さの校正がまだです。vision_measure.py で k を済ませてください")
    meas = Measurer(cfg)

    cap = None
    for idx in ([args.cam] if args.cam is not None else [1, 2, 3, 0]):
        cap = open_camera(idx)
        if cap:
            print(f"camera index {idx}")
            break
    if cap is None:
        raise SystemExit("カメラが開けません")

    port = find_arduino_port() if args.port == "auto" else args.port
    if port is None:
        raise SystemExit(1)
    try:
        grip = Gripper(port)
    except serial.SerialException as e:
        print(f"{port} を開けません: {e}")
        print("・Arduino IDEのシリアルモニタが開いていないか")
        print("・USBを抜き挿しした直後なら、数秒待ってから")
        find_arduino_port()
        raise SystemExit(1)
    args.port = port
    print(f"Arduino {args.port} 接続。物体: {args.obj}  設定の版: {args.tag}  記録先: {trials_csv}")
    print(__doc__.split("キー操作")[1].split("記録される")[0])

    win = "grasp_runner"
    cv2.namedWindow(win)
    cv2.createTrackbar("thresh", win, 40, 100, lambda v: None)

    cond = 3
    pending = None   # y/n 待ちの結果
    msg = "c: calibrate gripper -> o -> b: background"
    # 試行番号は、既存の trials.csv の続きから数える（起動し直しても番号が重複しない）
    counts = {}
    if trials_csv.exists():
        with open(trials_csv, newline="") as f:
            for row in csv.DictReader(f):
                k = (row.get("tag", "v1") or "v1", row["object"], row["condition"])
                counts[k] = max(counts.get(k, 0), int(row["trial_no"] or 0))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        thresh = max(cv2.getTrackbarPos("thresh", win), 1)
        r, mask, off = meas.measure(frame, thresh)
        view = frame.copy()
        if cfg["roi"]:
            x, y, w, h = cfg["roi"]
            cv2.rectangle(view, (x, y), (x + w, y + h), (255, 200, 0), 1)
        if mask is not None and pending is None:  # 物体と判定した部分を赤く重ねる
            rv = view[off[1]:off[1] + mask.shape[0], off[0]:off[0] + mask.shape[1]]
            rv[mask > 0] = (0.5 * rv[mask > 0] + [0, 0, 127]).astype(np.uint8)
        if r and pending is None:
            cv2.drawContours(view, [r["box"].astype(int)], 0, (0, 255, 0), 2)
            if r["width_mm"]:
                cv2.putText(view, f"cam W {r['width_mm']:.1f}mm", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        d = grip.get()
        cv2.putText(view, f"[{COND_LABEL[cond]}]  obj: {args.obj}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if d:
            cv2.putText(view, f"{d['state']}  wmm {d['wmm']:.1f}  fsr {int(d['fsrL'])}/{int(d['fsrR'])}"
                              f"  diff {int(d['diff'])}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(view, msg, (10, view.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            grip.send("c")
            msg = "calibrating... (wait #cal done)"
        elif key == ord("o"):
            grip.send("o")
            msg = "opened"
        elif key == ord("b"):
            capture_background(grip, cap, meas)
            msg = "background saved. place object, SPACE to grasp"
        elif key in (ord("1"), ord("2"), ord("3")):
            cond = key - ord("0")
            msg = f"condition {COND_LABEL[cond]}"
        elif key == ord(" ") and pending is None:
            if meas.bg is None and cond != 1:
                msg = "press b first (background)"
                continue
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw = logdir / f"raw_{stamp}_{COND_NAMES[cond]}.csv"
            grip.start_raw(raw)
            res, err = run_trial(cond, grip, cap, meas, thresh)
            grip.stop_raw()
            if err:
                msg = "ERROR"
                print(err)
                raw.unlink(missing_ok=True)
                continue
            res.update({"time": stamp, "tag": args.tag, "object": args.obj, "raw_file": raw.name})
            pending = res
            print(f"結果: {res['state']}  カメラ {res['cam_width_mm']}mm  指 {res['wmm']:.1f}mm  "
                  f"FSR {res['fsrL']}/{res['fsrR']}  {res['grasp_time_s']}s  → y:成功 n:失敗 x:捨てる")
            msg = f"{res['state']}  y: success / n: fail / x: discard"
        elif key in (ord("y"), ord("n")) and pending:
            k = (args.tag, args.obj, pending["condition"])
            counts[k] = counts.get(k, 0) + 1
            pending["trial_no"] = counts[k]
            pending["result"] = "success" if key == ord("y") else "fail"
            append_csv(trials_csv, pending)
            print(f"記録: {pending['condition']} #{counts[k]} {pending['result']}")
            msg = f"saved #{counts[k]} {pending['result']}. opening..."
            pending = None
            grip.send("o")
        elif key == ord("x") and pending:
            (logdir / pending["raw_file"]).unlink(missing_ok=True)
            pending = None
            grip.send("o")
            msg = "discarded. opening..."

    grip.send("o")
    grip.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
