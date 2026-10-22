/*
 * Wheel-speed encoder for the BFMC vehicle.
 *
 * Counts pulses from a hall-effect sensor on the drive wheel and reports the
 * resulting RPM over serial every REPORT_INTERVAL_MS. The Raspberry Pi reads
 * these lines (see src/bfmc/telemetry.py) and converts RPM to m/s using the
 * measured wheel circumference.
 *
 * Wiring: hall sensor signal -> D2 (interrupt-capable), VCC -> 5V, GND -> GND.
 */

const int HALL_PIN = 2;               // must be an interrupt-capable pin
const int PULSES_PER_REV = 2;         // magnets on the wheel
const unsigned long REPORT_INTERVAL_MS = 100;
const long BAUD_RATE = 115200;

volatile unsigned long pulse_count = 0;
unsigned long last_time = 0;

void onPulse() {
  pulse_count++;
}

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(HALL_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(HALL_PIN), onPulse, RISING);
  last_time = millis();
}

void loop() {
  delay(REPORT_INTERVAL_MS);

  unsigned long now = millis();
  unsigned long dt_ms = now - last_time;
  last_time = now;

  if (dt_ms == 0) {
    return;                           // guard against a millis() wrap
  }

  // Snapshot and clear the counter with interrupts off so a pulse arriving
  // mid-read is not lost.
  noInterrupts();
  unsigned long pulses = pulse_count;
  pulse_count = 0;
  interrupts();

  float revs = (float)pulses / PULSES_PER_REV;
  float rpm = revs / (dt_ms / 60000.0);

  Serial.println(rpm);
}
