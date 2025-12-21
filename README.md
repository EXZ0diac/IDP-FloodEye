# IDP FloodEye — Flood detection model and Telegram bot

This workspace contains two main artifacts:

- `model_training.py` — trains a flood-detection model using three sensors: rain sensor FR-04 (ADC 0–1023), ultrasonic sensor HC-SR04 (water level in cm), and waterflow sensor YF-S201 (flow in L/min). Saves a pipeline to `models/flood_pipeline.joblib`.
- `main.py` — a Telegram bot which accepts sensor values and returns a flood prediction.

Requirements
------------

Install the Python packages listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

Training the model
------------------

If you have a CSV dataset, place it at `data/flood_data.csv` with columns: `rain` (ADC 0–1023), `ultrasonic` (cm), `flow` (L/min), `flood` (where `flood` is 0 or 1). Then run:

```powershell
python model_training.py --data data/flood_data.csv --out models/flood_pipeline.joblib
```

If no CSV is present, `model_training.py` will synthesize a dataset, train a RandomForest model, print evaluation metrics, and save a pipeline in `models/flood_pipeline.joblib`.

Telegram bot
------------

1. Create a bot with @BotFather and obtain the bot token.
2. Set the environment variable `TELEGRAM_TOKEN` with your token (PowerShell):

```powershell
$env:TELEGRAM_TOKEN = "<your-token-here>"
```

3. Run the bot:

```powershell
python main.py
```

4. Send the bot a message with three numbers. Examples:

- `/predict 300 20 5`
- `300,20,5`
- `rain=300 ultrasonic=20 flow=5`

The bot will reply with a predicted probability and a Flood / No flood label.

Wiring / GPIO pins
------------------

The project uses Raspberry Pi BCM GPIO numbering by default. The wiring used
by the project (documented in `main.py`) is:

- GPIO 23 — water flow sensor (YF-S201)
- GPIO 17 — rain sensor (FR-04) (digital output)
- GPIO 2  — ultrasonic trigger (HC-SR04)
- GPIO 3  — ultrasonic echo (HC-SR04)

Notes:
- These pins are BCM numbers (not physical header pin numbers). If you prefer
	physical board pin numbers, convert accordingly when wiring.
- `main.py` only documents these constants and exposes a helper `get_gpio_pins()`;
	it intentionally avoids importing `RPi.GPIO` so it can run on non-Pi systems.
	If you want live sensor reads on a Raspberry Pi, I can add a small hardware
	module that uses `RPi.GPIO` with a simulator fallback for development.

Notes and next steps
--------------------

- The model and threshold are basic; for production you should gather labeled sensor data and re-train.
- Consider adding authentication or restricting which users can call the bot.
- Add logging, persistence, and a more robust API (webhook) if needed.

Travel-time estimator (heuristic)
---------------------------------

The bot also includes a small heuristic estimator that computes an approximate
time for a flood front to travel a given distance (the bot currently reports a
10 km estimate). This is a simple, non-validated hydraulic approximation intended
for quick, rough guidance only — it is NOT a replacement for proper flood
modelling or local authority warnings.

Key inputs and assumptions:

- `flow` is assumed to be the YF-S201 sensor reading in L/min. The estimator
	converts this to m^3/s.
- `ultrasonic` is assumed to be the water level reading in cm (converted to
	metres and used as an approximate flow depth). A minimum depth (5 cm) is
	enforced to avoid division-by-zero.
- A default channel width of 10 m is used to compute cross-sectional area; this
	is a configurable parameter in the function but not exposed in the bot by
	default.
- Velocity is estimated as v = Q / A (where Q is flow, A is width * depth).
- Conservative fallbacks are applied (minimum velocity ~0.01 m/s) so the
	estimator always returns a finite time rather than infinite or NaN.

Output and caveats:

- The estimator returns a human-readable travel time (hours/minutes/seconds)
	for the requested distance, or a short message if the time is extremely long
	(e.g., >1 year) or inputs are invalid.
- This approach ignores channel slope, roughness, storage, lateral inflows,
	detention basins, and urban drainage features; those factors strongly affect
	real flood wave propagation. Use professional hydraulic models for decisions.

If you want the estimator exposed or tuned (different default width, distance
parameter, or improved hydraulics), I can add command options or a simple test
page showing how the calculation behaves for different sensor values.

