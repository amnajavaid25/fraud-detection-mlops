from prometheus_client import start_http_server, Gauge
import random
import time

# Metrics
recall_metric = Gauge('fraud_recall', 'Fraud recall')
latency_metric = Gauge('api_latency', 'API latency')
drift_metric = Gauge('data_drift', 'Data drift')

# Start metrics server
start_http_server(8000)

print("Monitoring started on port 8000")

while True:
    recall = random.uniform(0.75, 0.98)
    latency = random.uniform(100, 500)
    drift = random.uniform(0.0, 1.0)

    recall_metric.set(recall)
    latency_metric.set(latency)
    drift_metric.set(drift)

    print(f"Recall={recall:.2f}, Latency={latency:.2f}, Drift={drift:.2f}")

    time.sleep(5)
