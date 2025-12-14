import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution
from experiment import *
from pprint import pprint
import matplotlib.pyplot as plt
from enum import Enum

class PlotColor(Enum):
    # 🔥 Primary (top performers)
    PRIMARY_1 = "#1f77b4"   # strong blue
    PRIMARY_2 = "#d62728"   # strong red

    # 🌫️ Secondary (muted, low emphasis)
    FADED_1 = "#aec7e8"     # light blue
    FADED_2 = "#ff9896"     # light red
    FADED_3 = "#c7c7c7"     # light gray
    FADED_4 = "#dbdb8d"     # light olive
    FADED_5 = "#9edae5"     # light cyan
    FADED_6 = "#f7b6d2"     # light pink
    FADED_7 = "#c5b0d5"     # light purple
    FADED_8 = "#c49c94"     # light brown
    FADED_9 = "#e7ba52"     # light yellow

def plot_result(x_indices, y_adaptive, sigma_val, color, linestyle, linewidth, zorder):
    # === PLOTTING ===
    if len(x_indices) > 0:
        plt.plot(x_indices, y_adaptive, label=f'c={sigma_val}', color=color.value, linestyle=linestyle, linewidth=linewidth, zorder=zorder)
        #pprint(x_indices)
        #pprint(y_adaptive)
    else:
        print(f"⚠️ No data processed for {name}.")


def evaluate_model(dataset_name, config, sigma_val: float) -> np.ndarray:
    x_indices, y_adaptive = run_benchmark_on_dataset(dataset_name, config, sigma_val)
    return x_indices, y_adaptive 


def aggregate(scores: np.ndarray) -> float:
    # "Whole line as high as possible" -> maximize average height
    return float(np.mean(scores))
    # alternatives:
    # return float(np.min(scores))           # worst-case
    # return float(np.quantile(scores, 0.1)) # robust low-end


if __name__ == "__main__":
    #pprint(DATASET_CONFIG)
    dataset_name, config = next(iter(DATASET_CONFIG.items()))
    #print(dataset_name,config)
    print(f"\n=== 🚀 PROCESSING: {dataset_name} ===")
    print(f"⚙️ Config: Fading(λ)={config['fading_factor']} | Horizon={config['horizon']}")


    study = optuna.create_study(direction="maximize")

    sigma_values = [0.1] + [0.5 * i for i in range(1, 11)]
    dist = {"c": CategoricalDistribution(sigma_values)}

    plt.figure(figsize=(10, 6))
    results = {}
    for sigma_val in sigma_values:  # strictly in order
        x_indices, scores = evaluate_model(dataset_name, config, sigma_val)
        #pprint(x_indices)
        #pprint(scores)
        results[sigma_val] = {
        "x": x_indices,
        "scores": scores,
        }
        obj_value = aggregate(scores)

        completed = optuna.trial.create_trial(
            params={"c": sigma_val},
            distributions=dist,
            value=obj_value,
        )
        study.add_trial(completed)

        print(f"c = {sigma_val:.1f}  objective = {obj_value:.6f}")

    print("\n=== RESULTS ===")
    print("Best c:", study.best_params["c"])
    print("Best objective:", study.best_value)


    first_two = 2
    colors = list(PlotColor)
    print("\nAll trials (sorted by objective, descending):")
    for t, color in zip(sorted(study.trials, key=lambda t: t.value, reverse=True), colors):
        print(f"c = {t.params['c']}  objective = {t.value:.6f}")
        sigma_val = t.params['c']
        if first_two == 2:
            plot_result(results[sigma_val]["x"], results[sigma_val]["scores"], sigma_val, color, linestyle='-', linewidth=3, zorder=11)
            first_two -= 1
        elif first_two == 1:   
            plot_result(results[sigma_val]["x"], results[sigma_val]["scores"], sigma_val, color, linestyle='-', linewidth=3, zorder=10)                 
            first_two -= 1
        else:
            plot_result(results[sigma_val]["x"], results[sigma_val]["scores"], sigma_val, color, linestyle='--',linewidth=1,zorder=1)


    #===Closing the plot===#
    plt.title(f'Sigma Coefficient Value Validation: {dataset_name} (λ={config["fading_factor"]})')
    plt.xlabel('Stream Position')
    plt.ylabel('NMI Score')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    safe_name = dataset_name.replace(" ", "_")
    filename = f"result_{safe_name}.png"
    plt.savefig(filename)
    print(f"✅ Saved Plot: {filename}")
    plt.close()