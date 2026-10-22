# Arduino encoder

`encoder/encoder.ino` reads a hall-effect wheel sensor and streams RPM over
USB serial at 115200 baud, one value per line, every 100 ms.

## Flashing

1. Open `encoder/encoder.ino` in the Arduino IDE (or `arduino-cli`).
2. Set `PULSES_PER_REV` to the number of magnets on your wheel.
3. Upload to the board, then connect it to the Raspberry Pi over USB.

## Checking it works

Find the port and watch the stream:

```bash
ls /dev/tty{USB,ACM}*
python -m serial.tools.miniterm /dev/ttyUSB0 115200
```

Set the port you found as `encoder.port` in your config. The Python side only
needs the wheel circumference (`wheel_circumference_m`, i.e. wheel diameter
times pi) — pulses per revolution is handled entirely on the Arduino.
