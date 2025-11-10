import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from scipy import stats

def generate_wafer_map(n_dies=400, pattern_type='random'):
    """Generate synthetic wafer test data with various defect patterns"""
    # Create wafer grid (20x20)
    x = np.arange(20)
    y = np.arange(20)
    xx, yy = np.meshgrid(x, y)
    
    # Start with random passing dice
    wafer_data = np.random.normal(100, 10, (20, 20))  # e.g., leakage current
    
    # Add different defect patterns
    if pattern_type == 'edge':
        # Edge ring pattern
        mask = (xx < 3) | (xx > 16) | (yy < 3) | (yy > 16)
        wafer_data[mask] += np.random.normal(50, 15, mask.sum())  # Higher leakage at edges
        
    elif pattern_type == 'cluster':
        # Random cluster defects
        center_x, center_y = np.random.randint(5, 15, 2)
        cluster_mask = ((xx - center_x)**2 + (yy - center_y)**2) < 16
        wafer_data[cluster_mask] += np.random.normal(80, 20, cluster_mask.sum())
        
    elif pattern_type == 'scratch':
        # Linear scratch pattern
        for i in range(10, 15):
            if i < 20:
                wafer_data[i, 5:15] += np.random.normal(60, 15, 10)
    
    # Create binary pass/fail (fail if leakage > threshold)
    threshold = 150
    wafer_binary = (wafer_data > threshold).astype(int)
    
    return xx, yy, wafer_data, wafer_binary

# Generate multiple wafer patterns
patterns = ['random', 'edge', 'cluster', 'scratch']
fig, axes = plt.subplots(2, 4, figsize=(15, 8))

for i, pattern in enumerate(patterns):
    xx, yy, wafer_data, wafer_binary = generate_wafer_map(pattern_type=pattern)
    
    # Plot raw data
    ax1 = axes[0, i]
    im1 = ax1.scatter(xx, yy, c=wafer_data, cmap='viridis', s=50)
    ax1.set_title(f'{pattern.title()} Pattern\nRaw Data')
    plt.colorbar(im1, ax=ax1)
    
    # Plot pass/fail with anomaly detection
    ax2 = axes[1, i]
    
    # Use Isolation Forest for anomaly detection
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    values = wafer_data.ravel()
    
    # Combine spatial and electrical data
    features = np.column_stack([coords, values])
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(features)
    anomalies = anomalies.reshape(20, 20)
    
    # Plot anomalies
    colors = ['green' if a == 1 else 'red' for a in anomalies.ravel()]
    ax2.scatter(xx, yy, c=colors, s=50)
    ax2.set_title(f'AI-Detected Anomalies')
    
    # Add clustering for pattern recognition
    if pattern != 'random':
        # Get coordinates of failed dice
        fail_coords = coords[wafer_binary.ravel() == 1]
        if len(fail_coords) > 1:
            dbscan = DBSCAN(eps=2.5, min_samples=2)
            clusters = dbscan.fit_predict(fail_coords)
            unique_clusters = np.unique(clusters)
            print(f"Pattern '{pattern}': Detected {len(unique_clusters) - ( -1 in clusters)} cluster(s)")

plt.tight_layout()
plt.show()