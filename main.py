"""
main.py

Telegram bot to receive sensor values and return a flood prediction.

Usage:
  1. Ensure you have a trained model at models/flood_pipeline.joblib (see model_training.py)
  2. Set environment variable TELEGRAM_TOKEN with your bot token
  3. Run: python main.py

Supported inputs:
  - /predict 300 20 5        -> rain ultrasonic flow
  - Send a message containing three numbers (comma/space separated): "300,20,5" or "rain=300 ultrasonic=20 flow=5"

Response:
  - Flood probability and binary prediction (Flood / No flood)

"""
import os
import re
import logging
import joblib
import numpy as np
from typing import Optional, Tuple, List
from math import isfinite
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import requests

# Config
load_dotenv()
ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, 'models', 'flood_pipeline.joblib')
TOKEN = os.environ.get('TELEGRAM_TOKEN')

# GPIO pin assignments (BCM numbering)
# - GPIO 23 : YF-S201 water flow sensor (digital pulse / counting)
# - GPIO 17 : FR-04 rain sensor (digital output / DO)
# - GPIO 2  : HC-SR04 ultrasonic trigger (TRIG)
# - GPIO 3  : HC-SR04 ultrasonic echo (ECHO)
# Note: We intentionally do not import RPi.GPIO here to avoid import errors on
# non-Raspberry-Pi development machines. These constants document the wiring
# and can be used by a separate hardware module that performs actual GPIO IO.
GPIO_PINS = {
    'flow': 18,
    'rain': 17,
    'trig': 23,
    'echo': 24,
}

def get_gpio_pins():
    """Return a copy of the GPIO pin mapping (BCM numbering).

    Use this from hardware-specific code to avoid hard-coding numbers in multiple places.
    """
    return dict(GPIO_PINS)

# CSV log path for sensor readings
DATA_DIR = os.path.join(ROOT, 'data')
LOG_CSV = os.path.join(DATA_DIR, 'sensor_log.csv')


def sensors_command(update: Update, context: CallbackContext):
    """Return recent sensor readings from the CSV log.

    Usage examples:
    - /sensors            -> last reading
    - /sensors 5          -> last 5 readings
    """
    # number of rows to return (default 1)
    n = 1
    if context.args:
        try:
            n = max(1, int(context.args[0]))
        except Exception:
            n = 1

    if not os.path.exists(LOG_CSV):
        update.message.reply_text('No sensor log found on device.')
        return

    # Read last n lines from CSV efficiently
    from collections import deque
    import csv

    try:
        with open(LOG_CSV, 'r', newline='', encoding='utf-8') as fh:
            dq = deque(fh, maxlen=n + 1)  # +1 in case header present
    except Exception as exc:
        update.message.reply_text(f'Error reading sensor log: {exc}')
        return

    if not dq:
        update.message.reply_text('Sensor log is empty.')
        return

    # Try to parse csv header to be friendly
    try:
        # If header exists, skip it
        lines = [l.strip() for l in dq if l.strip()]
        if not lines:
            update.message.reply_text('No readable data found in sensor log.')
            return
        if lines[0].lower().startswith('timestamp'):
            header = lines[0]
            data_lines = lines[1:]
        else:
            header = None
            data_lines = lines

        # Format reply
        reply_lines = []
        for ln in data_lines:
            reply_lines.append(ln)
        if not reply_lines:
            update.message.reply_text('No recent sensor rows found.')
            return
        text = 'Recent sensor readings:\n' + '\n'.join(reply_lines)
        update.message.reply_text(text)
    except Exception as exc:
        update.message.reply_text(f'Error processing sensor log: {exc}')


def status_command(update: Update, context: CallbackContext):
    """Fetch latest data from Raspberry Pi dashboard (must be reachable from this bot).

    Configure the dashboard URL with the environment variable `PI_DASHBOARD_URL`,
    e.g. http://192.168.1.50:5000
    """
    pi_url = os.environ.get('PI_DASHBOARD_URL')
    if not pi_url:
        update.message.reply_text('PI_DASHBOARD_URL not set on bot host.')
        return

    try:
        r = requests.get(pi_url.rstrip('/') + '/api/latest', timeout=5.0)
    except Exception as exc:
        update.message.reply_text(f'Error contacting Pi dashboard: {exc}')
        return

    if r.status_code != 200:
        update.message.reply_text(f'Pi dashboard returned {r.status_code}: {r.text[:200]}')
        return

    try:
        data = r.json()
    except Exception:
        update.message.reply_text('Could not parse response from Pi')
        return

    # Format a short, human-readable message
    lines = [f"Timestamp: {data.get('timestamp')}"]
    lines.append(f"Rain: {data.get('rain')}")
    lines.append(f"Ultrasonic (cm): {data.get('ultrasonic_cm')}")
    lines.append(f"Flow pulses: {data.get('pulses')} (pps={data.get('pulses_per_sec')})")
    lines.append(f"Flow L/min: {data.get('flow_lpm')}")
    update.message.reply_text('\n'.join(lines))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None


def load_model(path: str):
    global MODEL
    if MODEL is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Model not found at {path}. Please run model_training.py first.')
        MODEL = joblib.load(path)
        logger.info('Model loaded from %s', path)
    return MODEL


def parse_three_numbers(text: str) -> Optional[Tuple[float, float, float]]:
    """Extract three numeric values from text. Returns (rain, ultrasonic, flow) or None."""
    # find floats/ints
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) >= 3:
        try:
            a, b, c = float(nums[0]), float(nums[1]), float(nums[2])
            return a, b, c
        except ValueError:
            return None
    # try key=value style
    keys = {'rain': None, 'ultrasonic': None, 'flow': None}
    for k in keys:
        m = re.search(k + r"\s*=\s*(-?\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            keys[k] = float(m.group(1))
    if all(v is not None for v in keys.values()):
        return keys['rain'], keys['ultrasonic'], keys['flow']
    return None


def format_prediction(proba: float, pred: int) -> str:
    pct = proba * 100.0
    label = 'FLOOD' if pred == 1 else 'No flood'
    return f'Prediction: {label}\nProbability of flood: {pct:.1f}%'


def estimate_travel_time_km(distance_km: float, rain: float, ultrasonic: float, flow: float,
                            channel_width_m: float = 10.0) -> str:
    """Estimate time for a flood front to travel `distance_km` given sensors.

    This is a heuristic estimate using simplified hydraulics:
    
    - `ultrasonic` is water level in cm (INVERTED: 3cm=high water, 13cm=shallow)
      We map this to estimated depth (3cm→1m deep, 13cm→0m)
    - `flow` is pump sensor reading (0-12 L/min) used as flood intensity indicator
      We scale this to realistic flood velocity estimates (not absolute discharge)
    - Velocity ranges: Low ~0.5 m/s, Moderate ~2 m/s, High ~5 m/s for flood conditions
    
    Returns human-readable estimated travel time.
    """
    # Convert inputs
    try:
        # INVERTED scale: 3cm=high water, 13cm=shallow
        # Map: 3cm → 1.0m depth, 13cm → 0.0m depth
        ultrasonic_val = float(ultrasonic)
        depth_m = max(0.05, (16.0 - ultrasonic_val) / 10.0)
    except Exception:
        return 'Estimated travel time: unavailable (bad ultrasonic reading)'

    try:
        flow_val = float(flow)
    except Exception:
        return 'Estimated travel time: unavailable (bad flow reading)'

    # Use flow sensor (0-12 L/min) as intensity indicator
    # Map to realistic flood velocities:
    # 0-4 L/min (normal) → 0.5 m/s - 0.38 m/s
    # 5-8 L/min (moderate) → 2.0 m/s - 0.47, 0.66 m/s
    # >8 L/min (high) → 5.0 m/s - 0.75 m/s
    if flow_val <= 4:
        base_velocity = 0.5
    elif flow_val <= 8:
        # Linear interpolation: 4→0.5, 8→2.0
        base_velocity = 0.5 + (flow_val - 4) * (2.0 - 0.5) / 4.0
    else:
        # Linear interpolation: 8→2.0, 12→5.0
        base_velocity = 2.0 + (flow_val - 8) * (5.0 - 2.0) / 4.0
    
    # Adjust velocity based on depth (deeper water = faster flow)
    # depth 0.05m (shallow) → 0.6x, depth 1.0m (deep) → 1.5x
    depth_factor = 0.6 + (depth_m / 1.0) * 0.9
    velocity = base_velocity * depth_factor
    
    # Clamp to reasonable range
    velocity = max(0.3, min(velocity, 8.0))

    # Distance in metres
    distance_m = float(distance_km) * 1000.0

    time_s = distance_m / velocity

    # If it is raining, accelerate arrival estimate by 5 minutes to reflect wetter, faster conditions
    try:
        if float(rain) == 1:
            time_s = max(0.0, time_s - 5 * 60)
    except Exception:
        pass

    # If time is ridiculously large, mark as 'may not reach soon'
    hours = time_s / 3600.0
    if hours > 365 * 24:  # > 1 year
        return 'Estimated travel time: very long (>> 1 year) — flood unlikely to reach city based on current sensors.'

    # Format into hours/minutes
    h = int(hours)
    m = int((hours - h) * 60)
    s = int(time_s - (h * 3600 + m * 60))

    return f'Estimated travel time to city ({distance_km} km): {h}h {m}m {s}s (heuristic)'


def validate_sensor_values(rain: float, ultrasonic: float, flow: float) -> List[str]:
    """Return a list of warnings if sensor values are outside expected ranges.

    Expectations / recommended ranges:
    - rain: binary 0 (no rain) or 1 (rain)
    - ultrasonic: water level 3-13 cm (3cm=high water, 13cm=shallow)
    - flow: L/min 0-12 (12V pump max 12 L/min)

    These are soft checks: the function returns warnings but does not stop
    prediction. Adjust thresholds as needed for your deployment.
    """
    warnings: List[str] = []
    try:
        r = float(rain)
        if r not in (0, 1):
            warnings.append(f'rain value {r} should be 0 (no rain) or 1 (rain)')
    except Exception:
        warnings.append('rain value could not be parsed')

    try:
        u = float(ultrasonic)
        if u < 3:
            warnings.append(f'ultrasonic value {u} cm is below sensor range (3-13 cm)')
        elif u > 13:
            warnings.append(f'ultrasonic value {u} cm is above sensor range (3-13 cm)')
    except Exception:
        warnings.append('ultrasonic value could not be parsed')

    try:
        f = float(flow)
        if f < 0:
            warnings.append(f'flow value {f} L/min is negative')
        elif f > 12:
            warnings.append(f'flow value {f} L/min exceeds 12V pump max (12 L/min)')
    except Exception:
        warnings.append('flow value could not be parsed')

    return warnings


def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        'Hi — send /predict <rain> <ultrasonic> <flow> or send a message with three numbers (e.g. "300,20,5").'
    )


def help_cmd(update: Update, context: CallbackContext):
    update.message.reply_text(
        'Usage examples:\n/predict 300 20 5\nOr: 300,20,5\nOr: rain=300 ultrasonic=20 flow=5'
    )


def predict_command(update: Update, context: CallbackContext):
    text = ' '.join(context.args) if context.args else (update.message.text or '')
    parsed = parse_three_numbers(text)
    if parsed is None:
        update.message.reply_text(
            'Could not parse three numbers. Usage: /predict <rain> <ultrasonic> <flow>'
        )
        return
    rain, ultrasonic, flow = parsed
    model = load_model(MODEL_PATH)
    X = np.array([[rain, ultrasonic, flow]])
    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception:
        # If the model lacks predict_proba, fall back to predict and give binary
        pred = int(model.predict(X)[0])
        proba = float(pred)
    else:
        pred = int(proba >= 0.5)

    # Heuristic: ensure rain raises likelihood and no-rain slightly reduces it
    try:
        rain_flag = float(rain)
        if rain_flag == 1:
            proba = min(1.0, proba + 0.10)
        elif rain_flag == 0:
            proba = max(0.0, proba - 0.10)
        pred = int(proba >= 0.5)
    except Exception:
        pass

    # Estimate travel time to point B 1.5 km away and append to response (only if flood detected)
    reply_lines = [format_prediction(proba, pred)]
    if pred == 1:  # Only show estimation time if flood is detected
        time_text = estimate_travel_time_km(1.5, rain, ultrasonic, flow)
        reply_lines.append(time_text)
    reply = '\n'.join(reply_lines)
    update.message.reply_text(reply)


def message_handler(update: Update, context: CallbackContext):
    text = update.message.text or ''
    parsed = parse_three_numbers(text)
    if parsed is None:
        update.message.reply_text(
            'I could not find three numeric sensor values in your message.\nSend /help for examples.'
        )
        return
    rain, ultrasonic, flow = parsed
    model = load_model(MODEL_PATH)
    X = np.array([[rain, ultrasonic, flow]])
    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception:
        pred = int(model.predict(X)[0])
        proba = float(pred)
    else:
        pred = int(proba >= 0.5)

    # Heuristic: ensure rain raises likelihood and no-rain slightly reduces it
    try:
        rain_flag = float(rain)
        if rain_flag == 1:
            proba = min(1.0, proba + 0.10)
        elif rain_flag == 0:
            proba = max(0.0, proba - 0.10)
        pred = int(proba >= 0.5)
    except Exception:
        pass

    reply_lines = [format_prediction(proba, pred)]
    if pred == 1:  # Only show estimation time if flood is detected
        time_text = estimate_travel_time_km(1.5, rain, ultrasonic, flow)
        reply_lines.append(time_text)
    reply = '\n'.join(reply_lines)
    update.message.reply_text(reply)


def main():
    if TOKEN is None:
        raise EnvironmentError('Please set the TELEGRAM_TOKEN environment variable with your bot token.')

    # Preload model (optional)
    try:
        load_model(MODEL_PATH)
    except FileNotFoundError:
        logger.warning('Model not found; the bot will error on prediction until you run model_training.py')

    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(CommandHandler('help', help_cmd))
    dispatcher.add_handler(CommandHandler('predict', predict_command))
    dispatcher.add_handler(CommandHandler('sensors', sensors_command))
    dispatcher.add_handler(CommandHandler('status', status_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, message_handler))

    logger.info('Starting bot...')
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
