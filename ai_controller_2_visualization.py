#2. Articulating the Value: ROI & Business Case
#This module of the AI Controller present Quantifiable ROI Calculator for customers that adoptt the AI Controller
#The AI Controller module for this segment contains:
#Stacked ROI breakdown

#Sensitivity curve (ROI vs. yield improvement)

#Histogram of cost savings by component

#Pie chart for proportional impact.

import matplotlib.pyplot as plt
import numpy as np

class ROICalculator:
    def __init__(self, annual_volume, device_cost, test_time_hours, test_cost_per_hour):
        self.annual_volume = annual_volume
        self.device_cost = device_cost
        self.test_time_hours = test_time_hours
        self.test_cost_per_hour = test_cost_per_hour
        
    def calculate_savings(self, test_time_reduction, yield_improvement, false_failure_reduction):
        """Calculate comprehensive ROI"""
        # Test time savings
        current_test_cost = self.test_time_hours * self.test_cost_per_hour
        new_test_time = self.test_time_hours * (1 - test_time_reduction)
        test_time_savings = (current_test_cost - (new_test_time * self.test_cost_per_hour)) * self.annual_volume
        
        # Yield improvement savings
        current_yield_value = self.annual_volume * self.device_cost
        yield_savings = current_yield_value * yield_improvement
        
        # False failure reduction savings
        false_failure_savings = (self.annual_volume * false_failure_reduction * self.device_cost)
        
        total_savings = test_time_savings + yield_savings + false_failure_savings
        
        return {
            'test_time_savings': test_time_savings,
            'yield_savings': yield_savings, 
            'false_failure_savings': false_failure_savings,
            'total_savings': total_savings
        }

# Example calculation for AI accelerator customer
calculator = ROICalculator(
    annual_volume=1_000_000,      # 1M units
    device_cost=150,              # $150/device
    test_time_hours=8,            # 8 hours test time
    test_cost_per_hour=75         # $75/hour tester cost
)

savings = calculator.calculate_savings(
    test_time_reduction=0.25,     # 25% test time reduction
    yield_improvement=0.03,       # 3% yield improvement  
    false_failure_reduction=0.01  # 1% false failure reduction
)

print("ANNUAL ROI ANALYSIS")
print("==================")
for key, value in savings.items():
    print(f"{key:.<20} ${value:,.2f}")

# Create visualization
categories = ['Test Time', 'Yield Improvement', 'False Failure Reduction', 'Total']
values = [savings['test_time_savings'], savings['yield_savings'], 
          savings['false_failure_savings'], savings['total_savings']]

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
plt.title('Annual Savings from AI Controller Implementation\n(1M units @ $150/device)')
plt.ylabel('Savings ($)')
plt.xticks(rotation=45)

# Add value labels on bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000, 
             f'${value:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()






# # ==========================================================
# # ROI Visualization: Financial Impact of AI Controller
# # ==========================================================


# # Reuse ROI Calculator results from your code
categories = ['Test Time', 'Yield Improvement', 'False Failure Reduction']
values = [savings['test_time_savings'], savings['yield_savings'], savings['false_failure_savings']]

# --- Visualization 1: ROI component breakdown ---
plt.figure(figsize=(8,5))
plt.bar(categories, values, color=['#1E88E5', '#43A047', '#FB8C00'])
plt.title("AI Controller ROI Breakdown by Category")
plt.ylabel("Annual Savings ($)")
plt.grid(axis='y')
plt.show()

# --- Visualization 2: Pie chart ---
plt.figure(figsize=(6,6))
plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=120, colors=['#1E88E5', '#43A047', '#FB8C00'])
plt.title("Proportional ROI Contribution")
plt.show()

# --- Visualization 3: Sensitivity curve (ROI vs Yield Improvement) ---
yield_improvement_range = np.linspace(0, 0.05, 20)
total_savings = [calculator.calculate_savings(0.25, y, 0.01)['total_savings'] for y in yield_improvement_range]
plt.figure(figsize=(8,5))
plt.plot(yield_improvement_range*100, total_savings, 'o-', color='#6A1B9A')
plt.title("ROI Sensitivity to Yield Improvement")
plt.xlabel("Yield Improvement (%)")
plt.ylabel("Total Savings ($)")
plt.grid(True)
plt.show()