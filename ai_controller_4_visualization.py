# #Here, we can visualize the gains of partnerring with Nvidia and intergrating their GPU Acceleration platform such as Nvidia Run:ai 
# with Teradyne's AI Controller platform. We can visualzie Teradyne's AI Controller metrics such as
# Semiconductor Test Latency distribution
# Throughput trend vs. batch size
# GPU vs CPU comparison bar chart
# Result: GPU acceleration performance visualized clearly for management review.

# ==========================================================
# GPU Inference Benchmark Visualization
# ==========================================================
import numpy as np
import matplotlib.pyplot as plt

# Synthetic benchmark data
batch_sizes = np.array([100, 500, 1000, 2000, 5000])
gpu_times = np.array([0.05, 0.12, 0.20, 0.38, 0.90])
cpu_times = np.array([0.25, 0.80, 1.60, 3.50, 8.00])

throughput_gpu = batch_sizes / gpu_times
throughput_cpu = batch_sizes / cpu_times

# --- Visualization 1: Throughput comparison ---
plt.figure(figsize=(8,5))
plt.plot(batch_sizes, throughput_gpu, 'o-', label='GPU', color='#1E88E5')
plt.plot(batch_sizes, throughput_cpu, 's--', label='CPU', color='#E53935')
plt.title("Inference Throughput vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Devices / second")
plt.legend()
plt.grid(True)
plt.show()

# --- Visualization 2: Bar chart of total inference time ---
plt.figure(figsize=(8,5))
bar_width = 0.35
plt.bar(batch_sizes - 50, gpu_times, width=bar_width, label='GPU')
plt.bar(batch_sizes + 50, cpu_times, width=bar_width, label='CPU')
plt.title("Inference Time Comparison (GPU vs CPU)")
plt.xlabel("Batch Size")
plt.ylabel("Latency (seconds)")
plt.legend()
plt.grid(axis='y')
plt.show()