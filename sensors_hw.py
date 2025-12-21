"""sensors_hw.py

Hardware GPIO helpers with a simulator fallback for development on non-Pi
machines.

Functions
- read_sensors(duration=5, pulses_per_l=None, simulate_if_no_gpio=True)
    Returns a dict with keys: 'rain' (0/1), 'ultrasonic_cm' (float),
    'flow_lpm' (float or None), 'pulses' (int), 'pulses_per_sec' (float)

This module attempts to import RPi.GPIO. If unavailable, and
simulate_if_no_gpio is True, the functions return simulated values.
"""
from __future__ import annotations

import os
import time
import random
from typing import Optional, Dict

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except Exception:
    GPIO = None
    HAS_GPIO = False

# Try to import spidev for MCP3008 ADC (analog rain sensor)
try:
    import spidev
    HAS_SPI = True
except Exception:
    spidev = None
    HAS_SPI = False

# Default BCM pin mapping (kept here for convenience; main.py documents the
# mapping as well). Update if you change pins in `main.py`.
GPIO_PINS = {
    'flow': 18,
    # for analog rain on MCP3008 use ADC channel (see ADC_CHANNEL)
    'rain': 17,
    'trig': 23,
    'echo': 24,
}

# ADC configuration (MCP3008)
ADC_CHANNEL = 0
ADC_SPI_BUS = 0
ADC_SPI_DEVICE = 0

# Flow input pull direction (environment): 'UP' or 'DOWN' (default DOWN)
FLOW_PULL = os.environ.get('FLOW_PULL', 'DOWN').upper()
if FLOW_PULL not in ('UP', 'DOWN'):
    FLOW_PULL = 'DOWN'

# Map to RPi.GPIO constants when available
def _get_pull_const():
    if not HAS_GPIO:
        return None
    return GPIO.PUD_UP if FLOW_PULL == 'UP' else GPIO.PUD_DOWN

# If GPIO is present, disable warnings and try to cleanup any previous state so
# we don't get "channel is already in use" on re-runs.
if HAS_GPIO:
    try:
        GPIO.setwarnings(False)
    except Exception:
        pass
    try:
        GPIO.cleanup()
    except Exception:
        # cleanup may fail if not previously setup; ignore
        pass


def _simulate_reading(duration: float = 1.0) -> Dict[str, object]:
    pulses = int(random.uniform(0, 200))
    pulses_per_sec = round(pulses / max(1e-6, float(duration)), 3)
    # simulate analog rain 0..1023
    rain_analog = int(random.uniform(0, 1023))
    return {
        'rain': rain_analog,
        'ultrasonic_cm': round(random.uniform(5.0, 150.0), 1),
        'pulses': pulses,
        'pulses_per_sec': pulses_per_sec,
        'flow_lpm': round(random.uniform(0.0, 50.0), 2),
    }


def read_sensors(duration: float = 5.0,
                 pulses_per_l: Optional[float] = None,
                 simulate_if_no_gpio: bool = True,
                 adc_channel: int = ADC_CHANNEL) -> Dict[str, object]:
    """Read sensors and return a dict of values.

    Parameters
    - duration: seconds to sample the flow sensor pulses
    - pulses_per_l: if provided, convert counted pulses to L/min using
      flow_lpm = (pulses/duration)*60 / pulses_per_l
    - simulate_if_no_gpio: if True and RPi.GPIO is not importable, return
      simulated values instead of raising.

    Returns a dict with keys:
    - 'rain': 0 or 1 (digital)
    - 'ultrasonic_cm': measured distance in cm (float) or None on failure
    - 'pulses': number of pulses counted during `duration` (int)
    - 'pulses_per_sec': pulses / duration (float) or None
    - 'flow_lpm': computed L/min if pulses_per_l provided, else None
    """
    if not HAS_GPIO:
        if simulate_if_no_gpio:
            return _simulate_reading(duration)
        raise RuntimeError('RPi.GPIO not available')

    pins = GPIO_PINS
    flow_pin = pins['flow']
    rain_pin = pins['rain']
    trig = pins['trig']
    echo = pins['echo']

    # Setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(rain_pin, GPIO.IN)
    GPIO.setup(trig, GPIO.OUT)
    GPIO.setup(echo, GPIO.IN)
    GPIO.setup(flow_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    # Read rain: prefer analog reading from MCP3008 if SPI available
    rain_val = None
    if HAS_SPI:
        try:
            spi = spidev.SpiDev()
            spi.open(ADC_SPI_BUS, ADC_SPI_DEVICE)
            spi.max_speed_hz = 1350000
            # MCP3008 single-ended read
            adc = spi.xfer2([1, (8 + int(adc_channel)) << 4, 0])
            data = ((adc[1] & 3) << 8) + adc[2]
            rain_val = int(data)
            spi.close()
        except Exception:
            rain_val = None
    # If SPI/ADC not available, fall back to digital read if present
    if rain_val is None:
        try:
            rain_val = GPIO.input(rain_pin)
        except Exception:
            rain_val = None

    # Ultrasonic measure (HC-SR04)
    ultrasonic_cm: Optional[float] = None
    try:
        # Ensure trigger low
        GPIO.output(trig, False)
        time.sleep(0.05)

        # Send 10us pulse to trigger
        GPIO.output(trig, True)
        time.sleep(0.00001)
        GPIO.output(trig, False)

        # Wait for echo start
        start = time.time()
        timeout = start + 0.02
        while GPIO.input(echo) == 0 and time.time() < timeout:
            pulse_start = time.time()

        # Wait for echo end
        timeout2 = time.time() + 0.02
        while GPIO.input(echo) == 1 and time.time() < timeout2:
            pulse_end = time.time()

        # Compute distance
        if 'pulse_start' in locals() and 'pulse_end' in locals() and pulse_end > pulse_start:
            pulse_duration = pulse_end - pulse_start
            ultrasonic_cm = (pulse_duration * 34300.0) / 2.0
            ultrasonic_cm = round(ultrasonic_cm, 2)
    except Exception:
        ultrasonic_cm = None

    # Flow: count pulses for `duration` seconds. Prefer event_detect; fallback
    # to polling to detect rising edges if interrupts are not available.
    pulses_count = 0
    try:
        pulses = [0]

        def _cb(channel):
            pulses[0] += 1

        GPIO.add_event_detect(flow_pin, GPIO.RISING, callback=_cb)
        t0 = time.time()
        while time.time() - t0 < float(duration):
            time.sleep(0.1)
        pulses_count = int(pulses[0])
        try:
            GPIO.remove_event_detect(flow_pin)
        except Exception:
            pass
    except Exception:
        # fallback: simple polling edge detection
        try:
            t0 = time.time()
            prev = GPIO.input(flow_pin)
            pulses_local = 0
            while time.time() - t0 < float(duration):
                cur = GPIO.input(flow_pin)
                if prev == 0 and cur == 1:
                    pulses_local += 1
                prev = cur
                time.sleep(0.001)
            pulses_count = int(pulses_local)
        except Exception:
            pulses_count = 0
    pulses_per_sec = round(pulses_count / float(max(1e-6, duration)), 3)
    flow_lpm = None
    if pulses_per_l:
        try:
            flow_lpm = (pulses_per_sec * 60.0) / float(pulses_per_l)
            flow_lpm = round(flow_lpm, 3)
        except Exception:
            flow_lpm = None

    return {
        'rain': rain_val,
        'ultrasonic_cm': ultrasonic_cm,
        'pulses': pulses_count,
        'pulses_per_sec': round(pulses_per_sec, 3),
        'flow_lpm': flow_lpm,
    }
