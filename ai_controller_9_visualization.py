import pandas as pd
import matplotlib.pyplot as plt
# Simulated semiconductor test performance data
phases = ["Wafer Sort", "Final Test", "SLT"]
metrics = {
    "Baseline_Time_Hours": [8, 6, 5],
    "AI_Reduced_Time_Hours": [6, 4, 3],
    "Yield_Improvement_%": [3, 2, 1]
}
df_test = pd.DataFrame(metrics, index=phases)

# Bar + Line chart
fig, ax1 = plt.subplots(figsize=(8, 5))
df_test[["Baseline_Time_Hours", "AI_Reduced_Time_Hours"]].plot(kind='bar', ax=ax1)
ax1.set_ylabel("Test Time (Hours)")
ax1.set_title("AI-Driven Test Time Reduction Across Phases")

# Yield improvement overlay
ax2 = ax1.twinx()
ax2.plot(df_test.index, df_test["Yield_Improvement_%"], color="green", marker="o", linewidth=2)
ax2.set_ylabel("Yield Improvement (%)")

plt.tight_layout()
plt.show()


#Heatmap Correlations


import seaborn as sns

corr = df_test.corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="Blues")
plt.title("Feature Correlation Matrix: Test Time & Yield Improvement")
plt.show()