# ==========================================================
# 4.2 KPI Dashboard Visualization – AI Controller Program
# ==========================================================
# This module extends the KPI dashboard to include rich visuals
# highlighting progress against goals for Teradyne’s AI Test Controller.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------------------------------------
# Core KPI Dashboard Class
# ----------------------------------------------------------
class KPIDashboard:
    def __init__(self):
        self.kpis = {
            "business": {
                "ai_feature_attach_rate": {"target": 0.4, "current": 0.15},
                "software_arr_growth": {"target": 0.5, "current": 0.1},
                "customer_satisfaction_nps": {"target": 50, "current": 25}
            },
            "technical": {
                "test_time_reduction": {"target": 0.3, "current": 0.1},
                "inference_latency_ms": {"target": 10, "current": 25},
                "model_accuracy": {"target": 0.95, "current": 0.85}
            },
            "customer": {
                "adoption_rate": {"target": 0.6, "current": 0.2},
                "time_to_value_days": {"target": 30, "current": 60},
                "reference_customers": {"target": 10, "current": 2}
            }
        }
    
    def calculate_adoption_score(self):
        total_score = 0
        max_score = 0
        for category, metrics in self.kpis.items():
            for _, v in metrics.items():
                current, target = v["current"], v["target"]
                if target > 0:
                    score = min(current / target, 1.0) * 100
                    total_score += score
                    max_score += 100
        return (total_score / max_score) * 100 if max_score > 0 else 0
    
    def _format_kpi_section(self, category):
        section = ""
        for metric, v in self.kpis[category].items():
            current, target = v["current"], v["target"]
            progress = current / target if target > 0 else 0
            section += f"  {metric}: {current} / {target} ({progress:.1%})\n"
        return section
    
    def generate_quarterly_report(self):
        score = self.calculate_adoption_score()
        report = f"""
        TERADYNE AI CONTROLLER - QUARTERLY KPI REPORT
        {'=' * 50}
        Overall Adoption Score: {score:.1f}/100
        
        BUSINESS KPIs:
        {self._format_kpi_section('business')}
        
        TECHNICAL KPIs:
        {self._format_kpi_section('technical')}
        
        CUSTOMER KPIs:
        {self._format_kpi_section('customer')}
        """
        return report
    
    # ------------------------------------------------------
    # Visualization Section
    # ------------------------------------------------------
    def plot_dashboard(self):
        adoption_score = self.calculate_adoption_score()
        
        # Prepare data
        categories = []
        metrics = []
        current_values = []
        target_values = []
        
        for category, items in self.kpis.items():
            for metric, v in items.items():
                categories.append(category)
                metrics.append(metric)
                current_values.append(v["current"])
                target_values.append(v["target"])
        
        # Create DataFrame for plotting
        import pandas as pd
        df = pd.DataFrame({
            "Category": categories,
            "Metric": metrics,
            "Current": current_values,
            "Target": target_values
        })
        
        # ---- 1. KPI Progress by Category (Grouped Bar Chart)
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x="Metric", y="Target", hue="Category", palette="Blues", alpha=0.4)
        sns.barplot(data=df, x="Metric", y="Current", hue="Category", palette="crest", dodge=False)
        plt.xticks(rotation=45, ha="right")
        plt.title("KPI Progress by Category – Teradyne AI Controller", fontsize=14)
        plt.ylabel("Performance Value")
        plt.legend([],[], frameon=False)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
        # ---- 2. Adoption Score Gauge
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_xlim(0, 100)
        ax.barh([0], [adoption_score], color="seagreen", height=0.4)
        ax.barh([0], [100], color="lightgray", height=0.4, alpha=0.3)
        ax.text(adoption_score + 2, 0, f"{adoption_score:.1f}%", va="center", fontsize=12)
        ax.set_title("Overall Adoption Score", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        
        # ---- 3. Category-wise Performance Heatmap
        pivot = df.pivot_table(index="Category", columns="Metric", values="Current")
        plt.figure(figsize=(8, 4))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Category KPI Performance Heatmap")
        plt.tight_layout()
        plt.show()

# ----------------------------------------------------------
# Run & Display
# ----------------------------------------------------------
kpi_dashboard = KPIDashboard()
print(kpi_dashboard.generate_quarterly_report())
kpi_dashboard.plot_dashboard()
