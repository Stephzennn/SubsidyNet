import pandas as pd
import matplotlib.pyplot as plt
import os

base_dir = os.path.join(os.getcwd(), "results")

import pandas as pd
import matplotlib.pyplot as plt
import os

# Base path
base_dir = os.path.join(os.getcwd(), "results")

csv_names = [
    "epochs_vs_depth_80percenthe_normal",
    "epochs_vs_depth_glorot_normal",
    "epochs_vs_depth_glorot_uniform",
    "epochs_vs_depth_he_normal",
    "epochs_vs_depth_subsidy2_mds_glorot_uniform",
    "epochs_vs_depth_subsidy2_mds_he_uniform",
    "epochs_vs_depth_subsidyVersion4_mds_glorot_uniform"
]

init_map = {
    "glorot_uniform": "Glorot Uniform",
    "glorot_normal": "Glorot Normal",
    "he_normal": "He Normal",
    "he_uniform": "He Uniform"
}

def generate_title(name: str) -> str:
   
    if "subsidyVersion4" in name.lower():
        base = "Subsidy v4"
    elif "subsidy2" in name.lower():
        base = "Subsidy v2"
    elif "subsidy" in name.lower():
        base = "Subsidy"
    else:
        base = "Vanilla"

    #Determine initialization method
    for key, label in init_map.items():
        if key in name.lower():
            init = label
            break
    else:
        init = "Unknown Init"

    return f"{base} - {init}"

# Plot each CSV
for name in csv_names:
    csv_path = os.path.join(base_dir, f"{name}.csv")
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    if not {'depth', 'mean_epochs'}.issubset(df.columns):
        print(f"Missing expected columns in {csv_path}. Found: {df.columns.tolist()}")
        continue

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['depth'], df['mean_epochs'], marker='o', linestyle='-')
    plt.title(generate_title(name), fontsize=14)
    plt.xlabel("Depth")
    plt.ylabel("Epochs to 20%", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(base_dir, f"{name}_plot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")
