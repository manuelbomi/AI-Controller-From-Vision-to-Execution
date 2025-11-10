import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Simulated performance drift data

weeks = np.arange(1, 13)
baseline_f1 = 0.95 - np.random.rand(12)*0.05
retrained_f1 = 0.95 + np.random.rand(12)*0.02

plt.figure(figsize=(10,6))
plt.plot(weeks, baseline_f1, marker="o", label="Pre-Retraining")
plt.plot(weeks, retrained_f1, marker="o", label="Post-Retraining")
plt.xlabel("Week")
plt.ylabel("F1 Score")
plt.title("Automated Model Retraining: Performance Stabilization")
plt.legend()
plt.show()