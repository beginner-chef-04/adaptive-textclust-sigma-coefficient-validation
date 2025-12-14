import json
import os
import traceback
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
from river import cluster
from river import feature_extraction
from river import compose
from pprint import pprint
import river
# ==========================================
# 1. SCIENTIFIC CONFIGURATION
# ==========================================
# DATASET_CONFIG = {
#     "Tweets-T": {
#         "filename": "Tweets-T",
#         "fading_factor": 0.01,
#         "tgap": 200,
#         "horizon": 1000
#     },
#     "News-T": {
#         "filename": "News-T",
#         "fading_factor": 0.001,
#         "tgap": 200,
#         "horizon": 1000
#     },
#     "NT": {
#         "filename": "NT",
#         "fading_factor": 0.005,
#         "tgap": 200,
#         "horizon": 2000
#     },
#     "NTS": {
#         "filename": "NTS",
#         "fading_factor": 0.005,
#         "tgap": 200,
#         "horizon": 2000
#     },
#     "Trends-T": {
#         "filename": "Trends-T",
#         "fading_factor": 0.01,
#         "tgap": 200,
#         "horizon": 5000      
#     },
#     "SO-T": {
#         "filename": "SO-T",
#         "fading_factor": 0.001,
#         "tgap": 200,
#         "horizon": 5000
#     }
# }
DATASET_CONFIG = {
    "NT": {
        "filename": "NT",
        "fading_factor": 0.005,
        "tgap": 200,
        "horizon": 2000
    }
}


DATASETS_FOLDER = 'datasets' 

# ==========================================
# 2. PIPELINE BUILDER
# ==========================================
def create_pipeline(auto_r_status, config, sigma_val):
    return compose.Pipeline(
        feature_extraction.TFIDF(ngram_range=(1, 1)), 
        cluster.TextClust(
            real_time_fading=False, 
            auto_r=auto_r_status,
            radius=0.3,
            fading_factor=config["fading_factor"], 
            tgap=config["tgap"],
            sigma=sigma_val
        )
    )

# ==========================================
# 3. DATA LOADING
# ==========================================
def load_stream(filepath):
    if not os.path.exists(filepath):
        print(f"❌ Warning: File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            try:
                record = json.loads(line)
                # Try finding keys
                text = record.get('textCleaned') or record.get('body') or record.get('text') or record.get('title')
                label = record.get('clusterNo') or record.get('class') or record.get('label')
                
                if text and label is not None:
                    yield line_no, text, label
            except json.JSONDecodeError:
                continue 

# ==========================================
# 4. BENCHMARK ENGINE (ROBUST)
# ==========================================
def run_benchmark_on_dataset(name, config, sigma_val):
    raw_path = os.path.join(DATASETS_FOLDER, config["filename"])
    if os.path.exists(raw_path):
        filepath = raw_path
    elif os.path.exists(raw_path + ".json"):
        filepath = raw_path + ".json"
    else:
        print(f"❌ SKIPPING {name}: File not found.")
        return

    print(f"\n=== 🚀 PROCESSING: {name} ===")
    print(f"⚙️ Config: Fading(λ)={config['fading_factor']} | Horizon={config['horizon']}")

    #model_fixed = create_pipeline(False, config, sigma_val)
    model_adaptive = create_pipeline(True, config, sigma_val)
    #pprint(model_adaptive)
    #print(model_adaptive[-1].auto_r)
    #print(model_fixed[-1].auto_r)    
    #print(river.__version__)
    x_indices = []
    y_fixed = []
    y_adaptive = []
    
    buffer_true = []
    #buffer_pred_fixed = []
    buffer_pred_adaptive = []
    
    stream = load_stream(filepath)
    count = 0
    errors_caught = 0
    print(f'Sigma Value:{sigma_val}')

    for i, text, true_label in stream:
        count += 1
        
        try:
            # Prediction
            #p_fixed = model_fixed.predict_one(text)
            p_adaptive = model_adaptive.predict_one(text)

            # --- ROBUST LEARNING BLOCK ---
            #model_fixed.learn_one(text)
            model_adaptive.learn_one(text)
            
            buffer_true.append(true_label)
            #buffer_pred_fixed.append(p_fixed)
            buffer_pred_adaptive.append(p_adaptive)
            
        except ValueError as e:
            # Catch math domain errors (sqrt of negative number) and skip
            if "math domain error" in str(e):
                errors_caught += 1
                if errors_caught <= 5: # Only print the first few to avoid spam
                    print(f"⚠️ Warning: Math error at tweet {count}. Skipping observation.")
                continue
            else:
                raise e # Re-raise legitimate errors

        # --- Evaluation at Horizon ---
        if count % config["horizon"] == 0:
            #clean_fixed = [-1 if p is None else p for p in buffer_pred_fixed]
            clean_adaptive = [-1 if p is None else p for p in buffer_pred_adaptive]
            
            #nmi_f = normalized_mutual_info_score(buffer_true, clean_fixed)
            nmi_a = normalized_mutual_info_score(buffer_true, clean_adaptive)
            
            x_indices.append(count)
            #y_fixed.append(nmi_f)
            y_adaptive.append(nmi_a)
            #print(f'Sigma Value:{sigma_val}')
            #print(model_adaptive[-1].radius)
            #print(model_fixed[-1].radius)

            #print(f"   Step {count}: Adaptive={nmi_a:.3f}")
            
            buffer_true = []
            buffer_pred_fixed = []
            buffer_pred_adaptive = []


    return x_indices, y_adaptive

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- STARTING ROBUST MULTI-DATASET BENCHMARK ---")
    
    for dataset_name, config in DATASET_CONFIG.items():
        try:
            pprint(dataset_name)
            pprint(config)
            #run_benchmark_on_dataset(dataset_name, config)
        except Exception as e:
            print(f"💥 Unrecoverable Error on {dataset_name}: {e}")
            traceback.print_exc() # Print full error for debugging

    print("\n--- ALL BENCHMARKS FINISHED ---")