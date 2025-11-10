import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# 1. Generate correlated dataset across test stages
# -------------------------------------------------------------------
np.random.seed(42)
n_samples = 5000

# Wafer Sort parameters (early indicators)
ws_params = {
    'iddq_leakage': np.random.lognormal(2, 0.3, n_samples),
    'vdd_min': np.random.normal(1.0, 0.05, n_samples),
    'ring_osc_freq': np.random.normal(100, 10, n_samples),
}

# Final Test parameters
ft_params = {
    'max_freq': ws_params['ring_osc_freq'] * 0.9 + np.random.normal(0, 5, n_samples),
    'power_consumption': ws_params['iddq_leakage'] * 10 + np.random.normal(50, 10, n_samples),
    'thermal_resistance': np.random.normal(25, 3, n_samples),
}

# Hidden relationships to create SLT failure risk
slt_failure_risk = (
    0.3 * (ws_params['iddq_leakage'] > np.percentile(ws_params['iddq_leakage'], 90)) +
    0.4 * (ft_params['thermal_resistance'] > np.percentile(ft_params['thermal_resistance'], 85)) +
    0.3 * ((ws_params['vdd_min'] > 1.05) | (ws_params['vdd_min'] < 0.95)) +
    0.2 * (ft_params['max_freq'] < np.percentile(ft_params['max_freq'], 20))
)

# Convert to probability and binary outcomes
slt_failure_prob = 1 / (1 + np.exp(-(slt_failure_risk - 1.5 + np.random.normal(0, 0.2, n_samples))))
slt_failures = (slt_failure_prob > 0.5).astype(int)

print(f"SLT Failure Rate: {slt_failures.mean():.3f}")

# -------------------------------------------------------------------
# 2. Create combined dataset
# -------------------------------------------------------------------
feature_data = np.column_stack([
    ws_params['iddq_leakage'],
    ws_params['vdd_min'],
    ws_params['ring_osc_freq'],
    ft_params['max_freq'],
    ft_params['power_consumption'],
    ft_params['thermal_resistance']
])
feature_names = ['iddq_leakage', 'vdd_min', 'ring_osc_freq', 
                 'max_freq', 'power_consumption', 'thermal_resistance']

df = pd.DataFrame(feature_data, columns=feature_names)
df['slt_failure'] = slt_failures

# -------------------------------------------------------------------
# 3. Visualize failure distribution
# -------------------------------------------------------------------
plt.figure(figsize=(5, 4))
plt.pie(df['slt_failure'].value_counts(), labels=['Pass', 'Fail'], 
        autopct='%1.1f%%', colors=['#87CEFA', '#FF6347'])
plt.title('SLT Failure Rate Distribution')
plt.show()

# -------------------------------------------------------------------
# 4. Feature Correlation Heatmap
# -------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df[feature_names].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap (WS + FT Parameters)')
plt.show()

# -------------------------------------------------------------------
# 5. Train-test split and scaling
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    feature_data, slt_failures, test_size=0.3, random_state=42, stratify=slt_failures
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------------
# 6. Train Gradient Boosting model
# -------------------------------------------------------------------
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

# -------------------------------------------------------------------
# 7. Evaluate model
# -------------------------------------------------------------------
y_pred_proba = gb_model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

auc_score = roc_auc_score(y_test, y_pred_proba)
accuracy = (y_pred == y_test).mean()

print(f"\nSLT Failure Prediction Model Performance:")
print(f"AUC Score: {auc_score:.3f}")
print(f"Accuracy: {accuracy:.3f}")

# -------------------------------------------------------------------
# 8. Feature Importance Visualization
# -------------------------------------------------------------------
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance in SLT Failure Prediction')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 9. ROC Curve and Precision-Recall Curve
# -------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", color='blue')
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_title('ROC Curve')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend()

axes[1].plot(recall, precision, color='green')
axes[1].set_title('Precision–Recall Curve')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 10. Business Impact Analysis
# -------------------------------------------------------------------
test_cost_slt = 2.0  # $ per device for SLT
test_cost_ft = 0.5   # $ per device for Final Test

high_risk_mask = y_pred_proba > 0.7
low_risk_mask = y_pred_proba < 0.1

print(f"\nBusiness Impact Analysis:")
print(f"Devices identified as high risk: {high_risk_mask.mean():.1%}")
print(f"Devices identified as low risk: {low_risk_mask.mean():.1%}")

potential_savings = low_risk_mask.mean() * test_cost_slt
print(f"Potential cost savings per device: ${potential_savings:.2f}")

# -------------------------------------------------------------------
# 11. Business Risk vs Savings Visualization
# -------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(['High Risk Devices', 'Low Risk Devices'], 
        [high_risk_mask.mean()*100, low_risk_mask.mean()*100],
        color=['#E74C3C', '#2ECC71'])
plt.title('Risk Segmentation of Devices')
plt.ylabel('Percentage of Devices (%)')
plt.tight_layout()
plt.show()
