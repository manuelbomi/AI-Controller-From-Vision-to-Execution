#The AI Controller Partnership Strategy Visualization
#The Controller module in this segment can be used to show:
#Partnership integration roadmap (heatmap)

#Value by segment (bar chart)

#Collaboration intensity over time (line plot)

# ==========================================================
# Partnership Ecosystem Visualization
# ==========================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Partner': ['NVIDIA', 'Microsoft Azure', 'ANSYS'],
    'Phase 1': [3, 2, 1],
    'Phase 2': [4, 3, 2],
    'Phase 3': [5, 4, 3]
}
roadmap_df = pd.DataFrame(data).set_index('Partner')

# --- Visualization 1: Heatmap of integration progress ---
plt.figure(figsize=(6,4))
sns.heatmap(roadmap_df, annot=True, cmap='Blues', cbar_kws={'label': 'Integration Level'})
plt.title("Technology Partnership Maturity Heatmap")
plt.ylabel("Partner")
plt.show()

# --- Visualization 2: Partner value perception chart ---
partner_value = {
    'NVIDIA': 10,
    'Microsoft Azure': 8,
    'ANSYS': 7
}
plt.figure(figsize=(7,4))
plt.bar(partner_value.keys(), partner_value.values(), color=['#76B900','#0078D4','#FF9900'])
plt.title("Relative Partner Value Contribution")
plt.ylabel("Strategic Value Index")
plt.grid(axis='y')
plt.show()