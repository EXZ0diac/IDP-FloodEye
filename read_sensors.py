"""Simple CLI to print sensor readings to the terminal.

Usage examples:
  python read_sensors.py                # single reading (5s sample for flow)
  python read_sensors.py --duration 3   # sample flow for 3 seconds
  python read_sensors.py --watch 2      # print readings every 2 seconds
  python read_sensors.py --pulses-per-l 450
"""
from __future__ import annotations

import argparse
import time
from pprint import pprint

from sensors_hw import read_sensors


def main():
    p = argparse.ArgumentParser(description='Read sensors and print to terminal')
    p.add_argument('--duration', '-d', type=float, default=5.0,
                   help='Duration (s) to sample flow pulses')
    p.add_argument('--pulses-per-l', type=float, default=None,
                   help='Pulses per litre for your flow sensor (if known)')
    p.add_argument('--watch', '-w', type=float, default=None,
                   help='If provided, print readings every N seconds until Ctrl-C')
    p.add_argument('--no-simulate', action='store_true',
                   help="Don't simulate when RPi.GPIO is missing; raise instead")

    args = p.parse_args()

    try:
        if args.watch:
            print(f'Watching sensors every {args.watch} seconds (Ctrl-C to stop)')
            while True:
                vals = read_sensors(duration=args.duration,
                                    pulses_per_l=args.pulses_per_l,
                                    simulate_if_no_gpio=(not args.no_simulate))
                print('---')
                pprint(vals)
                time.sleep(args.watch)
        else:
            vals = read_sensors(duration=args.duration,
                                pulses_per_l=args.pulses_per_l,
                                simulate_if_no_gpio=(not args.no_simulate))
            pprint(vals)
    except KeyboardInterrupt:
        print('\nStopped by user')


if __name__ == '__main__':
    main()
