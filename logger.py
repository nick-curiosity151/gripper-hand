#!/usr/bin/env python3
"""Arduinoの出力をCSVに保存しつつ、キーボードからコマンドを送る。

使い方:
    pip install pyserial
    python3 logger.py /dev/ttyACM0        # Windowsなら COM3 など

入力例:  u1500 / a90 / o / g / s / c(校正) / q(終了)
ログ:    log_YYYYmmdd_HHMMSS.csv（PC受信時刻 + Arduinoの1行）
"""
import sys
import threading
import time
from datetime import datetime

import serial

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
ser = serial.Serial(port, 115200, timeout=1)
time.sleep(2)  # 接続時のリセット待ち
ser.write(b"r\n")  # データの出力を再開させる

fname = datetime.now().strftime("log_%Y%m%d_%H%M%S.csv")
running = True


def reader():
    with open(fname, "w") as f:
        f.write("pc_time,ms,us,fsrL,fsrR,fb,diff,wmm,state\n")
        last_show = 0.0
        while running:
            line = ser.readline().decode(errors="ignore").strip()
            if not line or line.startswith("ms,"):
                continue
            if line.startswith("#"):  # 校正などのメッセージは画面にだけ出す
                print("\n" + line)
                continue
            f.write(f"{time.time():.3f},{line}\n")
            if time.time() - last_show > 0.2:  # 画面表示は5Hzに間引き
                print("\r" + line.ljust(40), end="", flush=True)
                last_show = time.time()


th = threading.Thread(target=reader, daemon=True)
th.start()
print(f"logging -> {fname}   commands: u<us> / a<deg> / o / g / s / c / q")

try:
    while True:
        cmd = input().strip()
        if cmd == "q":
            break
        ser.write((cmd + "\n").encode())
finally:
    ser.write(b"p\n")  # 終了前にArduinoの出力を止める
    time.sleep(0.3)
    running = False
    time.sleep(0.2)
    ser.close()
    print(f"\nsaved: {fname}")
