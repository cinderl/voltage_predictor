from fastapi import FastAPI, Query
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
from math import pi
import os
from datetime import datetime
from fastapi.responses import PlainTextResponse

# --- Model Definition (identical to predict.py) ---
class TimeValuePredictor(nn.Module):
    def __init__(self, input_size, output_size=1, dropout_rate=0.2):
        super(TimeValuePredictor, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(input_size, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, output_size)
        # )
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # ADDED: Dropout layer 1
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # ADDED: Dropout layer 2
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )       
    def forward(self, x):
        return self.net(x)
# -------------------------------------------------

MODEL_PATH = './models/time_value_model.pth'
STATS_PATH = './models/normalization_stats.npz'
PROB_MODEL_PATH = './models/probability_model.pth'
PROB_STATS_PATH = './models/probability_stats.npz'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAGS = 0
INPUT_SIZE = 8+LAGS # hour_sin, hour_cos, minute_sin, minute_cos, weekday_sin, weekday_cos, month_sin, month_cos (+ lags)

app = FastAPI(title="PredictAPI", description="API for time value prediction", version="1.0")

class TimeValuePrediction(BaseModel):
    hour: int
    minute: int
    weekday: int
    month: int
    value: float

class PredictionResponse(BaseModel):
    current: TimeValuePrediction
    past_min: TimeValuePrediction
    future_min: TimeValuePrediction
    risk: float  # percentage risk (0..100) of anomaly at current time

# Load model and stats at startup
model = None
value_mean = None
value_std = None
prob_model = None
prob_low_threshold = None
prob_high_threshold = None

@app.on_event("startup")
def load_model_and_stats():
    global model, value_mean, value_std
    # Load normalization stats
    stats = np.load(STATS_PATH)
    value_mean = stats['mean'].item()
    value_std = stats['std'].item()
    # Load model
    model = TimeValuePredictor(INPUT_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    # Load probability model and thresholds if available
    global prob_model, prob_low_threshold, prob_high_threshold
    try:
        prob_stats = np.load(PROB_STATS_PATH)
        prob_low_threshold = float(prob_stats['low_threshold'].item())
        prob_high_threshold = float(prob_stats['high_threshold'].item())
    except Exception:
        prob_low_threshold = None
        prob_high_threshold = None
    try:
        prob_model = TimeValuePredictor(INPUT_SIZE).to(DEVICE)
        prob_model.load_state_dict(torch.load(PROB_MODEL_PATH, map_location=DEVICE))
        prob_model.eval()
    except Exception:
        prob_model = None


def adjust_time(hour: int, minute: int, delta_minutes: int) -> tuple[int, int]:
    """Adjust time by delta_minutes and return new (hour, minute)."""
    total_minutes = hour * 60 + minute + delta_minutes
    new_hour = (total_minutes // 60) % 24
    new_minute = total_minutes % 60
    return new_hour, new_minute


def adjust_time_with_weekday(hour: int, minute: int, weekday: int, delta_minutes: int) -> tuple[int, int, int]:
    """Adjust time and weekday by delta_minutes and return new (hour, minute, weekday).
       Weekday: Monday=0 .. Sunday=6
    """
    total_minutes = hour * 60 + minute + delta_minutes
    days_delta = total_minutes // (24 * 60)
    new_hour = (total_minutes // 60) % 24
    new_minute = total_minutes % 60
    new_weekday = (weekday + days_delta) % 7
    return new_hour, new_minute, new_weekday

def encode_time_and_lags(hour: int, minute: int, weekday: int | None = None, month: int | None = None, lags=None):
    h_sin = np.sin(2 * pi * hour / 24)
    h_cos = np.cos(2 * pi * hour / 24)
    m_sin = np.sin(2 * pi * minute / 60)
    m_cos = np.cos(2 * pi * minute / 60)
    if lags is None:
        lags = [0.0] * LAGS  # Default: zero lag features
    # Append weekday cyclical encoding if provided, otherwise default to Monday (0)
    if weekday is None:
        w_sin = 0.0
        w_cos = 1.0
    else:
        w_sin = np.sin(2 * pi * weekday / 7)
        w_cos = np.cos(2 * pi * weekday / 7)
    # Month cyclical encoding (month is 1..12). If not provided, use current month as fallback in caller.
    if month is None:
        # default month January -> month=1
        mo = 1
    else:
        mo = int(month)
    month_sin = np.sin(2 * pi * (mo - 1) / 12)
    month_cos = np.cos(2 * pi * (mo - 1) / 12)

    features = [h_sin, h_cos, m_sin, m_cos, w_sin, w_cos, month_sin, month_cos]
    features += lags
    return np.array([features], dtype=np.float32)

def get_prediction_for_time(hour: int, minute: int, weekday: int | None = None, month: int | None = None) -> TimeValuePrediction:
    """Get prediction for a specific hour, minute and optional weekday and month."""
    encoded_X = encode_time_and_lags(hour, minute, weekday, month)
    input_tensor = torch.tensor(encoded_X).to(DEVICE)
    with torch.no_grad():
        output_normalized = model(input_tensor).cpu().numpy()
    predicted_value = float((output_normalized[0][0] * value_std) + value_mean)
    # If weekday wasn't provided, default to 0 (Monday) - caller should pass correct weekday
    wd = weekday if weekday is not None else 0
    mo = month if month is not None else 1
    return TimeValuePrediction(hour=hour, minute=minute, weekday=wd, month=mo, value=predicted_value)


def get_risk_for_time(hour: int, minute: int, weekday: int | None = None, month: int | None = None) -> float:
    """Return percentage risk (0..100) of anomaly at given time using the probability model.
       If probability model not available, returns 0.0
    """
    global prob_model
    if prob_model is None:
        return 0.0
    encoded_X = encode_time_and_lags(hour, minute, weekday, month)
    input_tensor = torch.tensor(encoded_X).to(DEVICE)
    with torch.no_grad():
        logits = prob_model(input_tensor).cpu()
    # apply sigmoid to get probability
    probs = torch.sigmoid(logits).numpy()
    prob = float(probs[0][0])
    return prob * 100.0

def adjust_prediction_with_trend(current: TimeValuePrediction, 
                               past: TimeValuePrediction, 
                               future: TimeValuePrediction) -> TimeValuePrediction:
    """Adjust current prediction based on the trend between past and future values."""
    values = np.array([past.value, current.value, future.value])
    std_dev = np.std(values)
    
    # Calculate trend direction
    trend = future.value - past.value
    trend_direction = "INCREASING" if trend > 0 else "DECREASING" if trend < 0 else "STABLE"
    
    print("\n=== Voltage Trend Analysis ===")
    print(f"Past    ({past.hour:02d}:{past.minute:02d}): {past.value:.2f}V")
    print(f"Current ({current.hour:02d}:{current.minute:02d}): {current.value:.2f}V")
    print(f"Future  ({future.hour:02d}:{future.minute:02d}): {future.value:.2f}V")
    print(f"\nMetrics:")
    print(f"- Trend Direction: {trend_direction}")
    print(f"- Trend Magnitude: {abs(trend):.2f}V")
    print(f"- Standard Deviation: {std_dev:.2f}V")
    
    # Adjust current value based on trend direction and standard deviation
    trend_factor = 0.5  # Controls how much the trend influences the adjustment
    adjustment = (trend / 2) * (std_dev / np.abs(trend)) * trend_factor if trend != 0 else 0
    
    print(f"\nAdjustment:")
    print(f"- Factor: {trend_factor}")
    print(f"- Calculated Adjustment: {adjustment:.2f}V")
    print(f"- Original Value: {current.value:.2f}V")
    print(f"- Adjusted Value: {(current.value + adjustment):.2f}V")
    print("===========================\n")
    
    # Create new prediction with adjusted value
    return TimeValuePrediction(
        hour=current.hour,
        minute=current.minute,
        weekday=current.weekday,
        month=current.month,
        value=current.value + adjustment
    )

@app.get("/predict", response_model=PredictionResponse)
def predict(hour: int = Query(None, ge=0, le=23), minute: int = Query(None, ge=0, le=59), weekday: int = Query(None, ge=0, le=6), month: int = Query(None, ge=1, le=12)):
    # If hour or minute is not provided, use current time
    now = datetime.now()
    if hour is None:
        hour = now.hour
    if minute is None:
        minute = now.minute
    # If weekday not provided, use current weekday (Monday=0)
    if weekday is None:
        weekday = now.weekday()
    # If month not provided, use current month (1..12)
    if month is None:
        month = now.month
    
    if model is None or value_mean is None or value_std is None:
        return {"error": "Model or stats not loaded. Try again later."}


    
    # Get predictions for all time points
    past_hour, past_minute, past_weekday = adjust_time_with_weekday(hour, minute, weekday, -60)
    # Adjust month for past time if crossing month boundary is required: compute naive month shift using datetime
    try:
        from datetime import timedelta
        past_dt = datetime(now.year, month, hour, minute) + timedelta(minutes=-60)
        past_month = past_dt.month
    except Exception:
        past_month = month
    past_pred = get_prediction_for_time(past_hour, past_minute, past_weekday, past_month)

    future_hour, future_minute, future_weekday = adjust_time_with_weekday(hour, minute, weekday, 60)
    try:
        from datetime import timedelta
        future_dt = datetime(now.year, month, hour, minute) + timedelta(minutes=60)
        future_month = future_dt.month
    except Exception:
        future_month = month
    future_pred = get_prediction_for_time(future_hour, future_minute, future_weekday, future_month)

    # Get initial current prediction
    current_pred = get_prediction_for_time(hour, minute, weekday, month)
    
    # Adjust current prediction based on trend and standard deviation
    adjusted_current_pred = adjust_prediction_with_trend(current_pred, past_pred, future_pred)

    # Compute risk percentage for current time
    risk_pct = get_risk_for_time(hour, minute, weekday, month)

    return PredictionResponse(
        current=adjusted_current_pred,
        past_min=past_pred,
        future_min=future_pred
        , risk=risk_pct
    )

def encode_features(hour, minute, weekday):
    h_sin = np.sin(2 * pi * hour / 24)
    h_cos = np.cos(2 * pi * hour / 24)
    m_sin = np.sin(2 * pi * minute / 60)
    m_cos = np.cos(2 * pi * minute / 60)
    w_sin = np.sin(2 * pi * weekday / 7)
    w_cos = np.cos(2 * pi * weekday / 7)
    # For predictall, month is not iterated; assume January by default
    mo = 1
    month_sin = np.sin(2 * pi * (mo - 1) / 12)
    month_cos = np.cos(2 * pi * (mo - 1) / 12)
    features = [h_sin, h_cos, m_sin, m_cos, w_sin, w_cos, month_sin, month_cos]
    # Add lag features if needed
    if LAGS > 0:
        features += [0.0] * LAGS
    return np.array([features], dtype=np.float32)

@app.get("/predictall", response_class=PlainTextResponse)
def predict_all():
    import io
    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["weekday", "hour", "minute", "prediction", "risk_pct"])
    for weekday in range(7):
        for hour in range(24):
            for minute in range(60):
                encoded_X = encode_features(hour, minute, weekday)
                input_tensor = torch.tensor(encoded_X).to(DEVICE)
                with torch.no_grad():
                    output_normalized = model(input_tensor).cpu().numpy()
                predicted_value = (output_normalized * value_std) + value_mean
                # compute risk if probability model loaded
                try:
                    input_tensor_p = torch.tensor(encoded_X).to(DEVICE)
                    with torch.no_grad():
                        logits_p = prob_model(input_tensor_p).cpu()
                    prob_p = float(torch.sigmoid(logits_p).numpy()[0][0]) * 100.0
                except Exception:
                    prob_p = 0.0
                writer.writerow([weekday, hour, minute, float(predicted_value[0][0]), prob_p])
    return output.getvalue()
