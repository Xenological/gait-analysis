# Bilateral Gait Analysis System

Wearable system that measures walking patterns in real time using two ESP32 sensor units — one per foot. Built as a low-cost alternative to clinical gait analysis equipment (~$45 vs $1,000+).

## How It Works

Each foot unit has two FSRs (toe and heel pressure) and an MPU-6050 IMU. The ESP32 collects data at 30 Hz and streams it over Bluetooth to a laptop. A Python pipeline syncs both feet, filters the signals, detects heel strikes, and computes gait metrics.

## Metrics

- Cadence, stride length, stance/swing phase ratio
- Stride, stance, and cadence variability (CV%)
- Gait asymmetry index (left vs. right stride length difference)
- Double support time
- Jerk index (movement smoothness)
- Heel strike loading rate

## Usage

**Install dependencies**
```bash
pip install bleak pandas numpy matplotlib scipy scikit-learn
```

**Collect data**
```bash
python ble_receiver.py
```
Power on both sensor units, run the script, walk for 20–30 seconds, then press `Ctrl+C`. Two CSV files are saved automatically.

**Run analysis**
```bash
python processing_extraction.py
```
Picks up the latest CSV pair and prints results + plots.

## Files

- `ble_receiver.py` — BLE connection and CSV logging
- `processing_extraction.py` — full analysis pipeline

## Hardware

| Component | Qty |
|---|---|
| ESP32 dev board | 2 |
| FSR 402 | 4 |
| MPU-6050 IMU | 2 |
| Prototype PCB (hand-soldered) | 2 |

## Limitations

- Stride length from accelerometer integration drifts over time; partially corrected between heel strikes
- Assumes subject is standing still for the first second (IMU calibration)
- Tested on flat ground only
