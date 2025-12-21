import RPi.GPIO as GPIO
import time
import threading

# Pin setup
RAIN_PIN = 17
TRIG = 23
ECHO = 24
FLOW_PIN = 18

GPIO.setmode(GPIO.BCM)
# Avoid repeated warnings if pins were previously in use
try:
    GPIO.setwarnings(False)
except Exception:
    pass

# Try to cleanup previous state to avoid "channel already in use" errors
try:
    GPIO.cleanup()
except Exception:
    pass

GPIO.setup(RAIN_PIN, GPIO.IN)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(FLOW_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Flow sensor variables
flow_count = 0
flow_rate = 0.0
total_liters = 0.0
FLOW_CALIBRATION = 7.5  # Pulses per second per L/min (typical for YF-S201)

flow_event_supported = True

def flow_callback(channel):
    global flow_count
    flow_count += 1

# Try to register edge detection; if it fails, we'll fallback to polling in measure_flow
try:
    GPIO.add_event_detect(FLOW_PIN, GPIO.FALLING, callback=flow_callback)
except Exception as exc:
    print(f"Warning: could not add edge detection on pin {FLOW_PIN}: {exc}\nFalling back to polling mode.")
    flow_event_supported = False

def measure_flow():
    global flow_count, flow_rate, total_liters
    # If event detection failed, use polling with a short sample window
    if not flow_event_supported:
        while True:
            t0 = time.time()
            prev = GPIO.input(FLOW_PIN)
            pulses_local = 0
            while time.time() - t0 < 1.0:
                cur = GPIO.input(FLOW_PIN)
                if prev == 0 and cur == 1:
                    pulses_local += 1
                prev = cur
                time.sleep(0.001)
            pulse_diff = pulses_local
            flow_rate = (pulse_diff / FLOW_CALIBRATION)  # L/min
            total_liters += (flow_rate / 60.0)           # liters in 1 second
            print(f"[FLOW - POLL] Rate: {flow_rate:.2f} L/min | Total: {total_liters:.2f} L")
    else:
        while True:
            start_count = flow_count
            time.sleep(1)
            pulse_diff = flow_count - start_count
            flow_rate = (pulse_diff / FLOW_CALIBRATION)  # L/min
            total_liters += (flow_rate / 60.0)           # liters in 1 second
            print(f"[FLOW] Rate: {flow_rate:.2f} L/min | Total: {total_liters:.2f} L")

def measure_ultrasonic():
    GPIO.output(TRIG, False)
    time.sleep(0.5)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    duration = pulse_end - pulse_start
    distance = (duration * 34300) / 2  # cm
    return distance

def read_rain_sensor():
    if GPIO.input(RAIN_PIN) == 0:
        return "Rain Detected ☔"
    else:
        return "No Rain 🌤️"

try:
    print("🌧️ FloodEye System Started — Press CTRL+C to stop.")
    threading.Thread(target=measure_flow, daemon=True).start()

    while True:
        distance = measure_ultrasonic()
        rain_status = read_rain_sensor()

        print(f"[RAIN] {rain_status}")
        print(f"[ULTRASONIC] Water Level Distance: {distance:.2f} cm\n")

        time.sleep(2)

except KeyboardInterrupt:
    print("\nSystem Stopped by User.")
    GPIO.cleanup()
