"""sensor_logger.py

Continuously read sensors and append timestamped rows to a CSV file.

Usage:
  python sensor_logger.py               # run with defaults (5s interval, simulate if no GPIO)
  python sensor_logger.py --interval 2  # sample every 2 seconds
  python sensor_logger.py --out data/sensor_log.csv

Run this on the Raspberry Pi (use systemd or tmux to keep it running). The
script will create the data directory and append a header if the file doesn't
exist.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from typing import Optional

from sensors_hw import read_sensors, GPIO_PINS


def ensure_data_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def append_row(path: str, row: dict):
    header = ['timestamp', 'rain', 'ultrasonic_cm', 'pulses', 'pulses_per_sec', 'flow_lpm']
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def make_row(vals: dict) -> dict:
    return {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'rain': vals.get('rain'),
        'ultrasonic_cm': vals.get('ultrasonic_cm'),
        'pulses': vals.get('pulses'),
        'pulses_per_sec': vals.get('pulses_per_sec'),
        'flow_lpm': vals.get('flow_lpm'),
    }


def main():
    p = argparse.ArgumentParser(description='Continuously log sensor readings to CSV')
    p.add_argument('--interval', '-i', type=float, default=5.0,
                   help='Seconds between samples')
    p.add_argument('--pulses-per-l', type=float, default=None,
                   help='Convert flow pulses to L/min if known')
    p.add_argument('--out', type=str, default=os.path.join('data', 'sensor_log.csv'),
                   help='CSV output path (default data/sensor_log.csv)')
    p.add_argument('--simulate', action='store_true',
                   help='Force simulator mode even if RPi.GPIO is available')

    args = p.parse_args()

    ensure_data_dir(args.out)
    print(f'Logging sensor readings to {args.out} every {args.interval}s (simulate={args.simulate})')

    try:
        while True:
            vals = read_sensors(duration=1.0, pulses_per_l=args.pulses_per_l, simulate_if_no_gpio=args.simulate)
            row = make_row(vals)
            append_row(args.out, row)
            print(f"{row['timestamp']} -> rain={row['rain']} ultrasonic={row['ultrasonic_cm']}cm flow_lpm={row['flow_lpm']}")
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        print('\nStopped by user')


if __name__ == '__main__':
    main()
