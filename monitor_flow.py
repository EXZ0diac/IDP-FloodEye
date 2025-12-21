"""monitor_flow.py

Quick utility to test the YF-S201 flow sensor on a GPIO pin.

Features:
- Configurable BCM pin (env FLOW_PIN or --pin, default 18)
- Configurable pull (env FLOW_PULL or --pull, UP/DOWN, default DOWN)
- Counts pulses for a duration or watches continuously and prints rate
- Optional pulses-per-l to compute L/min
- Simulator fallback when RPi.GPIO not available for development/test

Examples:
  python monitor_flow.py --duration 10
  python monitor_flow.py --watch 2 --pulses-per-l 450
  FLOW_PULL=UP python monitor_flow.py --pin 18 --duration 5
"""
from __future__ import annotations

import argparse
import os
import time
import random
from typing import Optional

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except Exception:
    GPIO = None
    HAS_GPIO = False


def _simulate(duration: float) -> dict:
    pulses = int(random.uniform(0, 200))
    pps = round(pulses / max(1e-6, float(duration)), 3)
    return {'pulses': pulses, 'pulses_per_sec': pps}


def monitor_pin(pin: int = 18,
                duration: float = 5.0,
                pull: str = 'DOWN',
                pulses_per_l: Optional[float] = None,
                simulate: bool = False) -> dict:
    """Count pulses for `duration` seconds and return stats."""
    if simulate or not HAS_GPIO:
        return _simulate(duration)

    pull_const = GPIO.PUD_UP if pull == 'UP' else GPIO.PUD_DOWN
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.IN, pull_up_down=pull_const)

    # Try event detect first
    try:
        flow_lpm = [0]
        pulses = [0]

        def cb(ch):
            pulses[0] += 1

        GPIO.add_event_detect(pin, GPIO.RISING, callback=cb)
        t0 = time.time()
        while time.time() - t0 < float(duration):
            time.sleep(0.05)
        try:
            GPIO.remove_event_detect(pin)
        except Exception:
            pass
        count = int(pulses[0])
    except Exception:
        # Fallback to polling
        t0 = time.time()
        prev = GPIO.input(pin)
        count_local = 0
        while time.time() - t0 < float(duration):
            cur = GPIO.input(pin)
            if prev == 0 and cur == 1:
                count_local += 1
            prev = cur
            time.sleep(0.001)
        count = int(count_local)

    pps = round(count / max(1e-6, float(duration)), 3)
    flow_lpm = None
    if pulses_per_l:
        try:
            flow_lpm = round((pps * 60.0) / 450, 3)
        except Exception:
            flow_lpm = 0

    return {'pulses': count, 'pulses_per_sec': pps, 'flow_lpm': flow_lpm}


def main():
    p = argparse.ArgumentParser(description='Monitor flow sensor pulses on a GPIO pin')
    p.add_argument('--pin', type=int, default=int(os.environ.get('FLOW_PIN', 18)), help='BCM pin for flow sensor (default 18)')
    p.add_argument('--duration', '-d', type=float, default=5.0, help='Seconds to sample')
    p.add_argument('--watch', '-w', type=float, default=None, help='If set, repeat every N seconds')
    p.add_argument('--pull', choices=['UP', 'DOWN'], default=os.environ.get('FLOW_PULL', 'DOWN').upper(), help='Internal pull-up or pull-down')
    p.add_argument('--pulses-per-l', type=float, default=None, help='Sensor pulses per litre to compute L/min')
    p.add_argument('--simulate', action='store_true', help='Force simulator mode')
    args = p.parse_args()

    try:
        if args.watch:
            print(f'Watching pin BCM{args.pin} every {args.watch}s (pull={args.pull})')
            while True:
                stats = monitor_pin(pin=args.pin, duration=args.duration, pull=args.pull, pulses_per_l=args.pulses_per_l, simulate=args.simulate)
                print(time.strftime('%Y-%m-%d %H:%M:%S'), '->', stats)
                time.sleep(args.watch)
        else:
            stats = monitor_pin(pin=args.pin, duration=args.duration, pull=args.pull, pulses_per_l=args.pulses_per_l, simulate=args.simulate)
            print('Result:', stats)
    except KeyboardInterrupt:
        print('\nStopped by user')


if __name__ == '__main__':
    main()
