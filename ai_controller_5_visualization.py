# 4.1 Phased Adoption Strategy
# AI transformation in semiconductor testing requires different strategies across customer segments—from innovators to legacy fabs.
# The AdoptionStrategy class defines tailored go-to-market approaches, pilot programs, and incentives for each phase.

# ==========================================================
# 4.1 Phased Adoption Strategy – Visualization Enhanced
# ==========================================================
# This module illustrates how Teradyne’s AI Test Controller
# adoption expands across customer segments, pilot programs,
# and performance outcomes in a phased GTM approach.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# Core Strategy Class
# ----------------------------------------------------------
class AdoptionStrategy:
    def __init__(self):
        self.customer_segments = {
            "innovators": ["AI chip startups", "Advanced R&D groups"],
            "early_adopters": ["Tier 1 semiconductor", "Automotive leaders"], 
            "early_majority": ["Mainstream fabs", "Cost-sensitive customers"],
            "late_majority": ["Conservative test floors", "Legacy equipment users"]
        }
        
    def get_segment_strategy(self, segment):
        strategies = {
            "innovators": {
                "approach": "Technology partnership",
                "incentive": "Early access, co-development",
                "success_metric": "Reference customers"
            },
            "early_adopters": {
                "approach": "ROI-focused proof of concept", 
                "incentive": "Performance guarantees",
                "success_metric": "Production deployment"
            },
            "early_majority": {
                "approach": "Standardized solutions",
                "incentive": "Easy integration, training",
                "success_metric": "Volume adoption"
            },
            "late_majority": {
                "approach": "Risk reduction",
                "incentive": "Proven reliability, support",
                "success_metric": "Competitive parity"
            }
        }
        return strategies.get(segment, {})
    
    def create_pilot_program(self, customer_tier, use_case):
        """Design targeted pilot programs"""
        pilot_templates = {
            "test_time_reduction": {
                "duration": "30 days",
                "success_criteria": "15% test time reduction",
                "resources": "Dedicated AI engineer, dashboard access"
            },
            "yield_improvement": {
                "duration": "60 days", 
                "success_criteria": "2% yield improvement",
                "resources": "Data scientist, correlation tools"
            },
            "predictive_maintenance": {
                "duration": "90 days",
                "success_criteria": "50% reduction in unplanned downtime",
                "resources": "Monitoring tools, alert system"
            }
        }
        return pilot_templates.get(use_case, {})

# ----------------------------------------------------------
# Demonstration & Visualization
# ----------------------------------------------------------
strategy = AdoptionStrategy()

# ----- 1. Adoption Curve Simulation (Diffusion of Innovation)
segments = ["Innovators", "Early Adopters", "Early Majority", "Late Majority"]
adoption_rate = [5, 15, 45, 35]  # simulated adoption percentages

plt.figure(figsize=(8, 5))
sns.barplot(x=segments, y=adoption_rate, palette="crest")
plt.title("AI Controller Adoption Curve (Simulated)", fontsize=14)
plt.ylabel("Estimated % of Customer Base")
plt.xlabel("Customer Segment")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ----- 2. Pilot Program Emphasis (Use Case Focus)
use_cases = ["Test Time Reduction", "Yield Improvement", "Predictive Maintenance"]
program_durations = [30, 60, 90]
success_targets = [15, 2, 50]  # percent metrics

fig, ax1 = plt.subplots(figsize=(8, 5))
sns.barplot(x=use_cases, y=success_targets, palette="flare", ax=ax1)
ax1.set_ylabel("Target Improvement (%)")
ax1.set_xlabel("Pilot Program Focus")
plt.title("Pilot Program Success Targets", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ----- 3. Segment Maturity vs Success Metric (Heatmap)
data = pd.DataFrame({
    "Segment": segments,
    "Reference Customers": [1, 5, 10, 8],
    "Production Deployments": [0, 3, 7, 6],
    "Volume Adoptions": [0, 1, 8, 9]
}).set_index("Segment")

plt.figure(figsize=(7, 5))
sns.heatmap(data, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Customer Segment Maturity Metrics")
plt.xlabel("Key Success Indicators")
plt.ylabel("Customer Segment")
plt.tight_layout()
plt.show()
