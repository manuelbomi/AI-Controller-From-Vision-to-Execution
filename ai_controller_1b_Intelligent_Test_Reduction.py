import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------------------------------------------
# 1. Generate synthetic Final Test data
# -------------------------------------------------------------------
np.random.seed(42)
n_devices = 10000
n_tests = 50

data = {}
for i in range(n_tests):
    if i < 10:  # Core parameter tests - highly correlated
        data[f'test_{i}'] = np.random.normal(1.0, 0.1, n_devices)
    elif i < 30:  # Functional tests - moderately correlated
        data[f'test_{i}'] = np.random.normal(1.0, 0.2, n_devices) + 0.3 * data[f'test_{i-10}']
    else:  # Redundant/duplicate tests
        data[f'test_{i}'] = data[f'test_{i-25}'] + np.random.normal(0, 0.05, n_devices)

df = pd.DataFrame(data)
failures = np.random.choice([0, 1], size=n_devices, p=[0.95, 0.05])
df['final_result'] = failures

# Add noise
for col in df.columns[:-1]:
    df[col] += np.random.normal(0, 0.02, n_devices)

# -------------------------------------------------------------------
# 2. Basic dataset overview
# -------------------------------------------------------------------
print("Final Test Dataset Shape:", df.shape)
print("Failure Rate:", df['final_result'].mean())

# Visualize failure distribution
plt.figure(figsize=(5, 4))
sns.countplot(x='final_result', data=df, palette='coolwarm')
plt.title('Final Test Result Distribution')
plt.xlabel('Final Result (0 = Pass, 1 = Fail)')
plt.ylabel('Count')
plt.show()

# -------------------------------------------------------------------
# 3. Random Forest Feature Selection
# -------------------------------------------------------------------
X = df.drop('final_result', axis=1)
y = df['final_result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance visualization
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
importances.head(15).plot(kind='bar', color='teal')
plt.title('Top 15 Most Important Tests')
plt.ylabel('Feature Importance')
plt.xlabel('Test Feature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 4. Feature selection
# -------------------------------------------------------------------
selector = SelectFromModel(rf, prefit=True, threshold='median')
X_important = selector.transform(X)
selected_features = X.columns[selector.get_support()]

print(f"\nOriginal tests: {X.shape[1]}")
print(f"Selected tests: {len(selected_features)}")
print(f"Test reduction: {(1 - len(selected_features)/X.shape[1])*100:.1f}%")

# Visualize feature reduction
plt.figure(figsize=(6, 4))
plt.bar(['Original', 'Selected'], [X.shape[1], len(selected_features)], color=['#2E86AB', '#F18F01'])
plt.title('Feature Reduction')
plt.ylabel('Number of Tests')
plt.show()

# -------------------------------------------------------------------
# 5. Compare full vs reduced models
# -------------------------------------------------------------------
X_reduced = X[selected_features]
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

rf_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_reduced.fit(X_train_red, y_train_red)

full_acc = accuracy_score(y_test, rf.predict(X_test))
reduced_acc = accuracy_score(y_test_red, rf_reduced.predict(X_test_red))

print(f"\nFull model accuracy: {full_acc:.3f}")
print(f"Reduced model accuracy: {reduced_acc:.3f}")

# Accuracy comparison bar chart
plt.figure(figsize=(6, 4))
plt.bar(['Full Model', 'Reduced Model'], [full_acc, reduced_acc], color=['#009FB7', '#FED766'])
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.show()

# -------------------------------------------------------------------
# 6. Confusion Matrices for both models
# -------------------------------------------------------------------
y_pred_full = rf.predict(X_test)
y_pred_reduced = rf_reduced.predict(X_test_red)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_full), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix - Full Model')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test_red, y_pred_reduced), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Confusion Matrix - Reduced Model')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 7. Classification Report Summary
# -------------------------------------------------------------------
print("\nFull Model Classification Report:")
print(classification_report(y_test, y_pred_full))

print("Reduced Model Classification Report:")
print(classification_report(y_test_red, y_pred_reduced))
