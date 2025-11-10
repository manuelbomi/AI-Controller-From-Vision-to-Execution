import numpy as np
import matplotlib.pyplot as plt

tasks = ["Write Tests", "Debug", "Analyze", "Optimize", "Monitor"]
current = [40, 25, 15, 10, 10]
ai = [10, 5, 10, 5, 10]

plt.figure(figsize=(10,6))
x = np.arange(len(tasks))
plt.bar(x-0.2, current, width=0.4, label="Current Workflow")
plt.bar(x+0.2, ai, width=0.4, label="AI-Enhanced Workflow")
plt.xticks(x, tasks)
plt.ylabel("Hours per Week")
plt.title("Test Engineer Time Allocation: Before vs After AI")
plt.legend()
plt.show()


categories = ["Business", "Technical", "Customer"]
scores = [60, 70, 50]

plt.figure(figsize=(6,6))
plt.pie(scores, labels=categories, autopct="%1.1f%%", startangle=90, colors=["#0072B2", "#E69F00", "#56B4E9"])
plt.title("Overall KPI Distribution for AI Controller Adoption")
plt.show()