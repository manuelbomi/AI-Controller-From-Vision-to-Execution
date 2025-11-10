import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Synthetic roadmap data
data = {
    "Feature": [
        "Adaptive Test Flow", "Predictive Binning", "Real-time Yield Learning",
        "AI-driven Limit Optimization", "Fleet Quality Scoring"
    ],
    "Maturity": [70, 55, 40, 65, 50],
    "Impact": [90, 85, 80, 88, 75]
}

df = pd.DataFrame(data)

# Bar chart visualization
plt.figure(figsize=(10, 6))
plt.barh(df["Feature"], df["Maturity"], color="#0072B2", label="Feature Maturity")
plt.barh(df["Feature"], df["Impact"], alpha=0.5, color="#D55E00", label="Customer Impact")
plt.xlabel("Percent (%)")
plt.title("AI Controller Feature Roadmap: Maturity vs Customer Impact")
plt.legend()
plt.tight_layout()
plt.show()