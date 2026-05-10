from prometheus_client import start_http_server, Gauge
import random
import time

# -----------------------------
# SYSTEM METRICS
# -----------------------------
api_requests = Gauge('api_request_rate', 'API request rate')
api_latency = Gauge('api_latency', 'API latency')
error_rate = Gauge('error_rate', 'API error rate')
cpu_usage = Gauge('cpu_usage', 'CPU usage')
memory_usage = Gauge('memory_usage', 'Memory usage')

# -----------------------------
# MODEL METRICS
# -----------------------------
fraud_recall = Gauge('fraud_recall', 'Fraud recall')
false_positive_rate = Gauge('false_positive_rate', 'False positive rate')
prediction_confidence = Gauge('prediction_confidence', 'Prediction confidence')

# -----------------------------
# DATA METRICS
# -----------------------------
data_drift = Gauge('data_drift', 'Feature drift')
missing_values = Gauge('missing_values', 'Missing value ratio')
input_anomalies = Gauge('input_anomalies', 'Input anomalies')

# Start Prometheus exporter
start_http_server(8000)

print("Monitoring started on port 8000")

while True:

    # System metrics
    api_requests.set(random.uniform(100, 500))
    api_latency.set(random.uniform(100, 600))
    error_rate.set(random.uniform(0, 10))
    cpu_usage.set(random.uniform(20, 90))
    memory_usage.set(random.uniform(30, 95))

    # Model metrics
    fraud_recall.set(random.uniform(0.75, 0.98))
    false_positive_rate.set(random.uniform(0.01, 0.20))
    prediction_confidence.set(random.uniform(0.50, 1.00))

    # Data metrics
    data_drift.set(random.uniform(0.0, 1.0))
    missing_values.set(random.uniform(0.0, 0.5))
    input_anomalies.set(random.uniform(0, 10))

    print("Metrics updated")

    time.sleep(5)