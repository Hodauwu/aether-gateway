import requests
import pandas as pd
import time
import json
from tqdm import tqdm # Run 'pip install tqdm pandas requests' if needed

# API endpoint
URL = "http://localhost:8000/chat"

import pandas as pd

# Load your exact CSV
df = pd.read_csv("thesis_dataset_balanced.csv")

# Sample exactly 500 queries to save GPU time (250 of each label)
df = df.groupby('label').sample(n=250, random_state=42).sample(frac=1)

# Map the columns to match our API payload
test_queries = []
for _, row in df.iterrows():
    tier_string = "strong" if row["label"] == 1 else "cheap"
    test_queries.append({
        "prompt": row["text"],
        "true_tier": tier_string
    })

def run_benchmark(skip_cache: bool, output_filename: str):
    print(f"--- Starting Benchmark Run (skip_cache={skip_cache}) ---")
    results = []
    
    for idx, item in enumerate(tqdm(test_queries)):
        payload = {
            "message": item["prompt"],
            "tier": "auto",
            "skip_cache": skip_cache
        }
        
        start_time = time.time()
        try:
            response = requests.post(URL, json=payload)
            response.raise_for_status()
            data = response.json()
            
            end_time = time.time()
            total_roundtrip_latency = end_time - start_time
            
            # Extract Telemetry
            source = data.get("source", "unknown")
            router_tel = data.get("router_telemetry", {})
            llm_tel = data.get("llm_telemetry", {})
            
            results.append({
                "query_id": idx,
                "prompt": item["prompt"],
                "true_tier": item["true_tier"],
                "routed_tier": router_tel.get("routed_tier"),
                "router_confidence": router_tel.get("confidence", 1.0),
                "router_latency_ms": router_tel.get("router_latency_ms", 0.0),
                "source": source,
                "cache_distance": data.get("distance_metric", None),
                "llm_generation_time_s": llm_tel.get("generation_time_s", 0.0),
                "total_roundtrip_latency_s": total_roundtrip_latency,
            })
            
        except Exception as e:
            print(f"Error on query {idx}: {e}")
            
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_filename, index=False)
    print(f"Saved {len(df)} records to {output_filename}\n")

if __name__ == "__main__":
    # Run 1: The Baseline (Cache Disabled) - Measures raw LLM speed & Router accuracy
    run_benchmark(skip_cache=True, output_filename="benchmark_baseline.csv")
    
    # Run 2: The Optimized Gateway (Cache Enabled) - Measures semantic cache speedup
    run_benchmark(skip_cache=False, output_filename="benchmark_optimized.csv")
