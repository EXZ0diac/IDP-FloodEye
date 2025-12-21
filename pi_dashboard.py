"""pi_dashboard.py

Run this on the Raspberry Pi. It samples sensors every minute (uses
`sensors_hw.read_sensors`) and exposes a small Flask dashboard with JSON
endpoints so your laptop bot can fetch the latest reading.

Run:
  python3 pi_dashboard.py

The script writes `data/sensor_log.csv` and serves a web UI on port 5000.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
import csv
from typing import Deque, Dict, Any

from flask import Flask, jsonify, render_template_string, request

from sensors_hw import read_sensors
# Optional OpenCV webcam support
try:
  import cv2
  HAS_CV2 = True
except Exception:
  cv2 = None
  HAS_CV2 = False

# Simple thread-safe camera wrapper that reads frames using OpenCV
class Camera:
  def __init__(self, src=0, width=640, height=480, fps=15):
    self.src = src
    self.width = width
    self.height = height
    self.fps = fps
    self.capture = None
    self.frame = None
    self.running = False
    self.lock = threading.Lock()

  def start(self):
    if not HAS_CV2:
      return False
    if self.running:
      return True
    self.capture = cv2.VideoCapture(self.src)
    try:
      self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
      self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
      self.capture.set(cv2.CAP_PROP_FPS, self.fps)
    except Exception:
      pass
    self.running = True

    def _reader():
      while self.running:
        try:
          ret, frame = self.capture.read()
        except Exception:
          ret = False
          frame = None
        if not ret or frame is None:
          time.sleep(0.1)
          continue
        try:
          ret2, jpeg = cv2.imencode('.jpg', frame)
          if not ret2:
            continue
          with self.lock:
            self.frame = jpeg.tobytes()
        except Exception:
          pass
        time.sleep(1.0 / max(1, self.fps))

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return True

  def stop(self):
    self.running = False
    try:
      if self.capture:
        self.capture.release()
    except Exception:
      pass

  def get_frame(self):
    with self.lock:
      return self.frame

# Global camera instance
camera = Camera() if HAS_CV2 else None

# Config
ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT, 'data')
LOG_CSV = os.path.join(DATA_DIR, 'sensor_log.csv')
SAMPLE_INTERVAL = int(os.environ.get('SAMPLE_INTERVAL', '10'))  # seconds (default 10s for testing)
HISTORY_MAX = int(os.environ.get('HISTORY_MAX', '1440'))  # keep last 24h if 1m samples

app = Flask(__name__)

# In-memory deque of recent samples (dicts)
history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_MAX)


def ensure_data_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def append_row(path: str, row: Dict[str, Any]):
    header = ['timestamp', 'rain', 'ultrasonic_cm', 'pulses', 'pulses_per_sec', 'flow_lpm']
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def sample_and_store(pulses_per_l: float | None = None):
    """Background sampler: read sensors and append to CSV and in-memory history."""
    ensure_data_dir(LOG_CSV)
    while True:
        vals = read_sensors(duration=1.0, pulses_per_l=pulses_per_l, simulate_if_no_gpio=False)
        row = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'rain': vals.get('rain'),
            'ultrasonic_cm': vals.get('ultrasonic_cm'),
            'pulses': vals.get('pulses'),
            'pulses_per_sec': vals.get('pulses_per_sec'),
            'flow_lpm': vals.get('flow_lpm'),
        }
        append_row(LOG_CSV, row)
        history.append(row)
        print(f"{row['timestamp']} -> rain={row['rain']} ultrasonic={row['ultrasonic_cm']}cm flow_lpm={row['flow_lpm']}")
        time.sleep(SAMPLE_INTERVAL)


@app.route('/')
def index():
    # Simple page using Chart.js via CDN
  return render_template_string(INDEX_HTML, has_cv2=HAS_CV2)


@app.route('/api/latest')
def api_latest():
    if not history:
        return jsonify({'error': 'no data'}), 404
    return jsonify(history[-1])


@app.route('/api/history')
def api_history():
    n = int(request.args.get('n', 60))
    items = list(history)[-n:]
    return jsonify(items)


def _frame_generator():
  """Yield multipart JPEG frames for streaming via HTTP."""
  if not camera:
    # no camera available
    while True:
      time.sleep(1.0)
      yield b''
  while True:
    frame = camera.get_frame()
    if not frame:
      time.sleep(0.05)
      continue
    yield (b'--frame\r\n'
         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
  if not HAS_CV2 or not camera:
    return "Camera not available", 404
  return app.response_class(_frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


INDEX_HTML = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>FloodEye Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>body{font-family: Arial, sans-serif; margin:20px} .kv{margin-right:12px}</style>
  </head>
<body>
  <h1>FloodEye — Sensor Dashboard</h1>
  <div style="margin:12px 0">
    <strong>Live camera:</strong><br>
    {% if has_cv2 %}
      <img id="camera" src="/video_feed" width="640" height="480" alt="Camera feed">
    {% else %}
      <div style="color: #900">Camera support not available on this server.</div>
    {% endif %}
  </div>
  <div id="latestBlock">Latest: <span id="latest">loading...</span></div>
  <canvas id="chart" width="900" height="350"></canvas>
  <script>
    async function fetchData(){
      const res = await fetch('/api/history?n=120');
      const data = await res.json();
      return data;
    }

    function prepare(data){
      const labels = data.map(d=>d.timestamp);
      const ultrasonic = data.map(d=> (d.ultrasonic_cm==null? NaN : Number(d.ultrasonic_cm)) );
      const flow = data.map(d=> (d.flow_lpm==null? NaN : Number(d.flow_lpm)) );
      const pulses = data.map(d=> (d.pulses==null? NaN : Number(d.pulses)) );
      return {labels, ultrasonic, flow, pulses};
    }

    async function init(){
      const data = await fetchData();
      if(!data || data.length==0){ document.getElementById('latest').innerText='no data'; return }
      const p = prepare(data);
      const latest = data[data.length-1];
      document.getElementById('latest').innerText = `time: ${latest.timestamp}  |  ultrasonic: ${latest.ultrasonic_cm} cm  |  flow_lpm: ${latest.flow_lpm}  |  pulses: ${latest.pulses}`;

      const ctx = document.getElementById('chart').getContext('2d');
      const chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: p.labels,
          datasets: [
            { label: 'Ultrasonic (cm)', data: p.ultrasonic, borderColor: 'blue', fill:false, yAxisID:'y' },
            { label: 'Flow (L/min)', data: p.flow, borderColor: 'green', fill:false, yAxisID:'y' },
            { label: 'Pulses', data: p.pulses, borderColor: 'orange', fill:false, yAxisID:'y1' }
          ]
        },
        options: {
          responsive:true,
          interaction: {mode:'index', intersect:false},
          scales: {
            x: { display:true, title:{display:true, text:'Timestamp'}, ticks:{maxRotation:45, minRotation:0} },
            y: { display:true, title:{display:true, text:'Depth cm / L/min'} },
            y1: { display:true, position:'right', title:{display:true, text:'Pulses'}, grid:{drawOnChartArea:false} }
          }
        }
      });

      // refresh periodically
      setInterval(async ()=>{
        const d = await fetchData();
        if(d && d.length){
          const np = prepare(d);
          chart.data.labels = np.labels;
          chart.data.datasets[0].data = np.ultrasonic;
          chart.data.datasets[1].data = np.flow;
          chart.data.datasets[2].data = np.pulses;
          chart.update();
          const lt = d[d.length-1];
          document.getElementById('latest').innerText = `time: ${lt.timestamp}  |  ultrasonic: ${lt.ultrasonic_cm} cm  |  flow_lpm: ${lt.flow_lpm}  |  pulses: ${lt.pulses}`;
        }
      }, 60000);
    }
    init();
  </script>
</body>
</html>
'''


def main():
  # pulses_per_l: allow configuration via env, default to 450 if not set
  pulses_per_l = None
  try:
    env_ppl = os.environ.get('PULSES_PER_L')
    if env_ppl is None or env_ppl.strip() == '':
      pulses_per_l = 450.0
    else:
      val = float(env_ppl)
      pulses_per_l = val if val > 0 else 450.0
  except Exception:
    pulses_per_l = 450.0

  # Start sampler thread
  t = threading.Thread(target=sample_and_store, args=(pulses_per_l,), daemon=True)
  t.start()

  # Start camera if available
  if HAS_CV2 and camera:
    ok = camera.start()
    if ok:
      print('Camera started for /video_feed')
    else:
      print('Camera present but failed to start')

  # Run Flask
  app.run(host='0.0.0.0', port=int(os.environ.get('PI_DASH_PORT', '5000')), debug=False, threaded=True)


if __name__ == '__main__':
    main()
