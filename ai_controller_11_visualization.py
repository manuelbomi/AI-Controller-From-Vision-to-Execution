import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AdoptionStrategy:
    def __init__(self):
        self.customer_segments = {
            "innovators": ["AI chip startups", "Advanced R&D groups"],
            "early_adopters": ["Tier 1 semiconductor", "Automotive leaders"], 
            "early_majority": ["Mainstream fabs", "Cost-sensitive customers"],
            "late_majority": ["Conservative test floors", "Legacy equipment users"]
        }

# Visualization of adoption curve
segments = ["Innovators", "Early Adopters", "Early Majority", "Late Majority"]
adoption_pct = [2.5, 13.5, 34, 34]
plt.figure(figsize=(8,5))
plt.bar(segments, adoption_pct, color="#009E73")
plt.ylabel("Customer Base (%)")
plt.title("Technology Adoption Lifecycle for AI Controller")
plt.show()