import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Generate time-series data for probe card health monitoring
np.random.seed(42)
n_time_points = 200
n_sites = 48

# Create baseline contact resistance with gradual degradation
time = np.arange(n_time_points)
baseline_resistance = 1.0  # ohms

# Healthy period + degradation period
healthy_period = 100
degradation_start = 100

# Generate resistance data for all sites
resistance_data = np.zeros((n_time_points, n_sites))

for site in range(n_sites):
    # Base resistance with random variation per site
    site_variation = np.random.normal(0, 0.1)
    
    # Healthy period
    resistance_data[:healthy_period, site] = (
        baseline_resistance + 
        site_variation +
        np.random.normal(0, 0.05, healthy_period)
    )
    
    # Degradation period - different failure modes
    if site < 10:  # Sites with gradual degradation
        degradation = np.linspace(0, 0.8, n_time_points - healthy_period)
    elif site < 15:  # Sites with step degradation (contamination)
        degradation = np.ones(n_time_points - healthy_period) * 0.6
    elif site < 20:  # Sites with increasing variance (loose probe)
        degradation = np.random.normal(0, np.linspace(0.1, 0.5, n_time_points - healthy_period))
    else:  # Remaining sites stay relatively healthy
        degradation = np.random.normal(0, 0.05, n_time_points - healthy_period)
    
    resistance_data[healthy_period:, site] = (
        baseline_resistance + 
        site_variation + 
        degradation
    )

# Calculate site-to-site variation (key health metric)
site_variation = resistance_data.std(axis=1)
mean_resistance = resistance_data.mean(axis=1)

# Anomaly detection using One-Class SVM
features = np.column_stack([mean_resistance, site_variation])

# Train on first 80 points (known good period)
train_features = features[:80]

oc_svm = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)
oc_svm.fit(train_features)

# Predict anomalies on full dataset
anomalies = oc_svm.predict(features)
anomaly_scores = oc_svm.decision_function(features)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Mean resistance and variation
ax1.plot(time, mean_resistance, label='Mean Resistance', color='blue', linewidth=2)
ax1.fill_between(time, 
                mean_resistance - site_variation, 
                mean_resistance + site_variation, 
                alpha=0.3, label='±1σ Variation')
ax1.axvline(x=degradation_start, color='red', linestyle='--', label='Degradation Start')
ax1.set_ylabel('Resistance (Ω)')
ax1.set_title('Probe Card Health Monitoring')
ax1.legend()
ax1.grid(True)

# Plot 2: Anomaly scores
ax2.plot(time, anomaly_scores, color='green', linewidth=2)
ax2.axhline(y=0, color='red', linestyle='--', label='Anomaly Threshold')
ax2.fill_between(time, anomaly_scores, 0, where=(anomaly_scores<0), 
                color='red', alpha=0.3, label='Anomaly Region')
ax2.set_ylabel('Anomaly Score')
ax2.set_title('AI Anomaly Detection')
ax2.legend()
ax2.grid(True)

# Plot 3: Failure prediction
healthy_mask = anomalies == 1
ax3.plot(time[healthy_mask], mean_resistance[healthy_mask], 
        'go', label='Healthy', alpha=0.7)
ax3.plot(time[~healthy_mask], mean_resistance[~healthy_mask], 
        'ro', label='Anomaly', alpha=0.7)
ax3.set_xlabel('Time (Test Cycles)')
ax3.set_ylabel('Mean Resistance (Ω)')
ax3.set_title('Probe Card Health Classification')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

# Calculate maintenance alerts
first_anomaly = np.where(~healthy_mask)[0]
if len(first_anomaly) > 0:
    first_anomaly_time = first_anomaly[0]
    print(f"First anomaly detected at cycle: {first_anomaly_time}")
    print(f"Maintenance alert triggered {degradation_start - first_anomaly_time} cycles BEFORE major degradation")