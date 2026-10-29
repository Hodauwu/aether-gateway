import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the generated datasets
df_base = pd.read_csv("benchmark_baseline.csv")
df_opt = pd.read_csv("benchmark_optimized.csv")

# Set up the visual style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

# ==========================================
# 1. LATENCY CDF (Cumulative Distribution)
# Proves how much faster the semantic cache is
# ==========================================
plt.figure(figsize=(8, 6))
sns.ecdfplot(data=df_base, x="total_roundtrip_latency_s", label="Baseline (Live LLM Only)", linewidth=2.5)
sns.ecdfplot(data=df_opt, x="total_roundtrip_latency_s", label="Aether Gateway (With Cache)", linewidth=2.5, linestyle="--")
plt.title("CDF of End-to-End Latency")
plt.xlabel("Latency (Seconds)")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.xlim(0, df_base["total_roundtrip_latency_s"].quantile(0.95)) # Zoom in on the 95th percentile
plt.savefig("thesis_latency_cdf.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 2. CONFUSION MATRIX (Router Accuracy)
# Proves DistilBERT routes correctly
# ==========================================
plt.figure(figsize=(6, 5))
cm = confusion_matrix(df_base["true_tier"], df_base["routed_tier"], labels=["cheap", "strong"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["cheap", "strong"], yticklabels=["cheap", "strong"])
plt.title("DistilBERT Router Confusion Matrix")
plt.xlabel("Predicted Tier (Routed)")
plt.ylabel("Actual Tier (True)")
plt.savefig("thesis_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 3. EXPECTED CALIBRATION ERROR (ECE)
# Proves the Softmax probabilities are trustworthy
# ==========================================
plt.figure(figsize=(8, 6))
# Calculate if prediction was correct
df_base['is_correct'] = (df_base['true_tier'] == df_base['routed_tier']).astype(int)
# Group into 10 confidence bins
bins = np.linspace(0.5, 1.0, 11)
df_base['conf_bin'] = pd.cut(df_base['router_confidence'], bins=bins)
calibration = df_base.groupby('conf_bin', observed=False).agg(
    accuracy=('is_correct', 'mean'),
    confidence=('router_confidence', 'mean'),
    count=('is_correct', 'count')
).dropna()

plt.plot([0.5, 1.0], [0.5, 1.0], 'k--', label="Perfectly Calibrated")
plt.plot(calibration['confidence'], calibration['accuracy'], marker='o', linewidth=2.5, label="DistilBERT Router")
plt.title("Reliability Diagram (Calibration)")
plt.xlabel("Router Confidence (Softmax)")
plt.ylabel("Actual Accuracy")
plt.legend()
plt.savefig("thesis_ece_calibration.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 4. PARETO FRONTIER (Cost vs. Latency)
# Proves the overall architectural efficiency
# ==========================================
plt.figure(figsize=(8, 6))
base_latency = df_base["total_roundtrip_latency_s"].mean()
base_llm_calls = len(df_base)

opt_latency = df_opt["total_roundtrip_latency_s"].mean()
opt_llm_calls = len(df_opt[df_opt["source"] == "live_llm"])

plt.scatter([base_llm_calls], [base_latency], color='red', s=200, label='Baseline (Monolithic)')
plt.scatter([opt_llm_calls], [opt_latency], color='green', s=200, label='Aether Gateway (Optimized)')
plt.plot([base_llm_calls, opt_llm_calls], [base_latency, opt_latency], 'k--', alpha=0.5)

plt.title("System Efficiency (Pareto Frontier)")
plt.xlabel("Total Expensive LLM Invocations (Compute Cost)")
plt.ylabel("Average System Latency (Seconds)")
plt.legend()
plt.savefig("thesis_pareto.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Successfully generated all 4 thesis graphs!")
