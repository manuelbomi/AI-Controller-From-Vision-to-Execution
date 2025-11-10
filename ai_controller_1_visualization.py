# ==========================================================
# AI Controller: Real-Time Test Data Visualization
# ==========================================================
#This part of the AI Controller produces four plots:
#Signal drift line plot

#Voltage histogram with anomalies

#Feature correlation heatmap

#Anomaly proportion pie chart

# This simulates a real-time analytics dashboard in Phase 1 ("AI Assistant") for early fault detection and process drift visualization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# --- Generate synthetic semiconductor test data (1000 devices) ---
np.random.seed(42)
device_data = pd.DataFrame({
    'device_id': np.arange(1000),
    'voltage': np.random.normal(1.0, 0.05, 1000),
    'current': np.random.normal(5.0, 0.3, 1000),
    'temperature': np.random.normal(25, 2, 1000),
    'frequency': np.random.normal(2.4, 0.1, 1000)
})

# Inject a few anomalies
for col in ['voltage', 'current']:
    device_data.loc[np.random.choice(1000, 15), col] *= np.random.uniform(1.2, 1.5)

# --- Anomaly detection using Isolation Forest ---
model = IsolationForest(contamination=0.02, random_state=42)
device_data['anomaly_score'] = model.fit_predict(device_data[['voltage', 'current', 'temperature', 'frequency']])
device_data['is_anomaly'] = device_data['anomaly_score'] == -1

# --- Visualization 1: Real-time signal drift ---
plt.figure(figsize=(10,5))
plt.plot(device_data['device_id'], device_data['voltage'], label='Voltage (V)')
plt.plot(device_data['device_id'], device_data['current'], label='Current (A)')
plt.title("Device-Level Electrical Signal Drift Across Batch")
plt.xlabel("Device ID")
plt.ylabel("Measurement Value")
plt.legend()
plt.grid(True)
plt.show()

# --- Visualization 2: Anomaly distribution ---
plt.figure(figsize=(6,5))
plt.hist(device_data['voltage'][~device_data['is_anomaly']], bins=30, alpha=0.7, label='Normal')
plt.hist(device_data['voltage'][device_data['is_anomaly']], bins=30, alpha=0.7, label='Anomaly')
plt.title("Voltage Distribution with Anomalies Highlighted")
plt.xlabel("Voltage (V)")
plt.ylabel("Device Count")
plt.legend()
plt.show()

# --- Visualization 3: Correlation heatmap ---
plt.figure(figsize=(6,5))
sns.heatmap(device_data[['voltage','current','temperature','frequency']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap - Wafer Sort Data")
plt.show()

# --- Visualization 4: Anomaly ratio pie chart ---
labels = ['Normal Devices', 'Anomalies']
sizes = [len(device_data) - device_data['is_anomaly'].sum(), device_data['is_anomaly'].sum()]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#4CAF50','#FF7043'])
plt.title("Anomaly Detection Summary")
plt.show()