// gripper_test.ino  —  UNO R4 WiFi
// サーボ（グリッパー）＋左右FSR＋位置フィードバックの動作確認用
//
// 配線
//   サーボ信号(白) -> D9
//   サーボ電源(赤/黒) -> 外部電源（黒はArduinoのGNDとも共通にする）
//   サーボのフィードバック線(緑) -> A2（約0.5V=全開 〜 2.2V=全閉）
//   FSR左 : 5V -- FSR -- A0 -- 10kΩ -- GND
//   FSR右 : 5V -- FSR -- A1 -- 10kΩ -- GND
//
// シリアル(115200)コマンド  ※改行で確定
//   u<マイクロ秒>  例: u1500  パルス幅で直接指定（細かく動かせる。約0.5ms=全開〜2.2ms=全閉）
//   a<角度>        例: a90    角度で指定（a0 = 544us、a180 = 2400us）
//   o              開く（PULSE_OPEN）
//   g              把持：FSRがしきい値を超えるまで少しずつ閉じて止める
//   s              停止（現在位置で保持）
//   p              データの出力を止める（PC側のプログラムが終了する前に送る）
//   r              データの出力を再開する（PC側のプログラムが起動したときに送る）
//   c              自動校正：何も挟まずに実行。全開→全閉へ100usずつ動かしてfbを記録し、
//                  diff/wmmの計算に使う（約10秒。電源を入れ直したら毎回やると良い）
//
// 出力（50Hz, CSV）: ms,us,fsrL,fsrR,fb,diff,wmm,state
//   us   = 指令パルス幅[マイクロ秒]
//   fb   = フィードバック電圧[mV]（実際の指の位置）
//   diff = us - (fbから推定した位置[us])。自由に動いている間はほぼ0。
//          指が物体や機構の端に当たって止まると、閉じる方向でプラスに大きくなる
//          fb→位置の換算は3点校正（折れ線）：CAL_US[] と CAL_FB[] の組
//   wmm  = 実際の指の位置(fb)から推定した指先の隙間[mm]。物をつかんだら≒物の幅

#include <Servo.h>

// ===== 校正で書き換える値（単位はマイクロ秒）=====
const int PULSE_MIN   = 560;   // 指令できる下限（機構の限界より少し内側に）
const int PULSE_MAX   = 2150;  // 指令できる上限
const int PULSE_OPEN  = 560;   // 全開
const int PULSE_CLOSE = 2120;  // 何も挟まないときの全閉（把持時もこれ以上閉じない）
const int FSR_TOUCH   = 80;    // 接触判定（0〜1023）
const int FSR_HOLD    = 400;   // 把持完了判定
const int STALL_DIFF  = 80;    // diffがこれを超えたら「指が物に当たって止まった」と判定
const int STALL_BACK  = 30;    // 止まったとき、実際の指の位置からこれだけ指令を戻す（押し付け防止）
const int STEP_US     = 10;    // 把持時に1回で閉じる量
const int STEP_MS     = 30;    // 把持時に閉じる間隔
// fbの2点校正（自由に動いている状態で実測した us と fb の組）
// fbの校正表（自由に動いている状態の us と fb の組。小さい順）
// 初期値は手で測った3点。c コマンドで細かい表に置き換わる
const int CAL_MAX = 24;
int   CAL_N = 3;
float CAL_US[CAL_MAX] = {560, 1500, 2150};
float CAL_FB[CAL_MAX] = {485, 1383, 2002};

// 指先の隙間の実測（us と mm の組。小さい順）
const int   WID_N = 5;
const float WID_US[WID_N] = {560,  1000, 1500, 2000, 2150};
const float WID_MM[WID_N] = {27.7, 19.0, 11.4, 4.6,  3.2};
// ===============================================

const int PIN_SERVO = 9;
const int PIN_FSR_L = A0;
const int PIN_FSR_R = A1;
const int PIN_FB    = A2;

Servo servo;
int us = PULSE_OPEN;
enum State { IDLE, GRASPING, HOLDING, STALL, LIMIT } state = IDLE;
const char* STATE_NAME[] = {"IDLE", "GRASPING", "HOLDING", "STALL", "LIMIT"};
//   HOLDING = 左右のFSRがどちらもしきい値を超えて停止（触覚で把持）
//   STALL   = FSRは片側しか反応しないが、指が物に当たって止まったので停止（位置で把持）
//   LIMIT   = 全閉まで閉じても何にも当たらなかった（空振り）

unsigned long lastStep = 0, lastPrint = 0;
bool streaming = true;  // false の間はCSVを出さない（USBの橋渡しが詰まるのを防ぐ）
String buf;

// fb[mV] → 位置[us] に換算（折れ線補間）
float fbToUs(float fb) {
  int i = 0;
  while (i < CAL_N - 2 && fb > CAL_FB[i + 1]) i++;
  float d = CAL_FB[i + 1] - CAL_FB[i];
  if (d < 1) d = 1;  // 0で割らない
  return CAL_US[i] + (fb - CAL_FB[i]) * (CAL_US[i + 1] - CAL_US[i]) / d;
}

// 位置[us] → 指先の隙間[mm]（折れ線補間）
float usToMm(float u) {
  int i = 0;
  while (i < WID_N - 2 && u > WID_US[i + 1]) i++;
  return WID_MM[i] + (u - WID_US[i]) * (WID_MM[i + 1] - WID_MM[i]) / (WID_US[i + 1] - WID_US[i]);
}

void moveUs(int p) {
  us = constrain(p, PULSE_MIN, PULSE_MAX);
  servo.writeMicroseconds(us);
}

long readFbAvg(int n) {
  long sum = 0;
  for (int i = 0; i < n; i++) { sum += analogRead(PIN_FB); delay(2); }
  return sum * 5000L / 16383 / n;
}

// 自動校正：開いた側から一方向に動かして記録（ギアのガタの影響をそろえる）
void calibrate() {
  Serial.println("#cal start");
  moveUs(PULSE_MIN);
  delay(800);
  float tmpUs[CAL_MAX], tmpFb[CAL_MAX];
  int n = 0;
  for (int p = PULSE_MIN; n < CAL_MAX; p += 100) {
    if (p > PULSE_MAX) p = PULSE_MAX;
    moveUs(p);
    delay(400);
    tmpUs[n] = p;
    tmpFb[n] = readFbAvg(32);
    Serial.print("#cal,"); Serial.print(p); Serial.print(','); Serial.println(tmpFb[n], 0);
    n++;
    if (p == PULSE_MAX) break;
  }
  moveUs(PULSE_OPEN);

  // 結果の確認：指が全開→全閉まで動いたなら、fbは1000mV以上変わり、単調に増えるはず
  bool ok = (tmpFb[n - 1] - tmpFb[0]) > 1000;
  for (int i = 1; i < n && ok; i++) if (tmpFb[i] <= tmpFb[i - 1]) ok = false;
  if (!ok) {
    Serial.println("#cal FAILED: 指が動いていません。サーボの電源(電池)と配線を確認してください。前の校正表のまま使います");
    return;
  }
  for (int i = 0; i < n; i++) { CAL_US[i] = tmpUs[i]; CAL_FB[i] = tmpFb[i]; }
  CAL_N = n;
  Serial.println("#cal done");
}

void handleCommand(String c) {
  c.trim();
  if (c.length() == 0) return;
  char k = c.charAt(0);
  if (k == 'u')      { state = IDLE; moveUs(c.substring(1).toInt()); }
  else if (k == 'a') { state = IDLE; moveUs(map(c.substring(1).toInt(), 0, 180, 544, 2400)); }
  else if (k == 'o') { state = IDLE; moveUs(PULSE_OPEN); }
  else if (k == 'g') { state = GRASPING; }
  else if (k == 's') { state = IDLE; }
  else if (k == 'c') { state = IDLE; calibrate(); }
  else if (k == 'p') { streaming = false; }
  else if (k == 'r') { streaming = true; Serial.println("ms,us,fsrL,fsrR,fb,diff,wmm,state"); }
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(14);  // UNO R4は14bit(0〜16383)で読める。fbを細かく測るため
  // 既定(544〜2400us)より広い範囲を指令できるようにする
  servo.attach(PIN_SERVO, 400, 2500);
  moveUs(PULSE_OPEN);
  Serial.println("ms,us,fsrL,fsrR,fb,diff,wmm,state");
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') { handleCommand(buf); buf = ""; }
    else buf += ch;
  }

  int fL = analogRead(PIN_FSR_L) >> 4;            // FSRは従来どおり0〜1023に換算
  int fR = analogRead(PIN_FSR_R) >> 4;
  long fb = analogRead(PIN_FB) * 5000L / 16383;   // mV
  float fbUs = fbToUs(fb);
  long diff = us - (long)fbUs;
  float wmm = usToMm(fbUs);
  unsigned long now = millis();

  // 把持：左右どちらもFSR_HOLDを超えたら停止。
  //       指が物に当たって止まったら(diff大)、少し戻して停止。全閉に達したら打ち切り
  if (state == GRASPING && now - lastStep >= STEP_MS) {
    lastStep = now;
    if (fL >= FSR_HOLD && fR >= FSR_HOLD) state = HOLDING;
    else if (diff >= STALL_DIFF) { moveUs((int)fbUs + STALL_BACK); state = STALL; }
    else if (us >= PULSE_CLOSE)           state = LIMIT;
    else moveUs(us + STEP_US);
  }

  if (streaming && now - lastPrint >= 20) {
    lastPrint = now;
    Serial.print(now); Serial.print(',');
    Serial.print(us);  Serial.print(',');
    Serial.print(fL);  Serial.print(',');
    Serial.print(fR);  Serial.print(',');
    Serial.print(fb);  Serial.print(',');
    Serial.print(diff); Serial.print(',');
    Serial.print(wmm, 1); Serial.print(',');
    Serial.println(STATE_NAME[state]);
  }
}
