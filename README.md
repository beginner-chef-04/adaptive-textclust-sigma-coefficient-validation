
# Adaptive textClust Validation
### Validating the Robustness of a Two-Phase Manipulation Detection Framework

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/River-Online%20ML-green)
![Status](https://img.shields.io/badge/Status-Academic%20Validation-orange)

## 📖 About The Project

This project is a **critical empirical validation** conducted as part of the Master's Seminar on *Textual Data Streams and Social Media Analytics* at the **University of Paderborn**.

The study synthesizes two foundational papers by **Assenmacher & Trautmann** to validate a robust pipeline for detecting manipulation campaigns in social media streams:

1.  **The Application:** A Two-Phase Framework for Detecting Manipulation Campaigns (2020).
2.  **The Engine:** Textual One-Pass Stream Clustering with Automated Distance Threshold Adaptation (2022).

### 🎯 Research Goal
The goal is to empirically demonstrate that the **Automated Distance Threshold Adaptation**  introduced in the 2022 methodology is a prerequisite for the reliability of the 2020 Framework. We compare the **Adaptive** algorithm against a **Fixed Threshold** baseline across multiple datasets to measure stability under high-velocity concept drift.

---

## ⚙️ Technical Approach

The pipeline is implemented using **[River](https://riverml.xyz/)**, a Python library for online machine learning.

* **Algorithm:** `textClust` (Stream Clustering).
* **Feature Extraction:** Incremental TF-IDF (1-grams).
* **Evaluation Metric:** Interval-based Normalized Mutual Information (NMI) to account for temporal evolution.
* **Datasets Benchmarked:**
    * **Social Streams:** `Tweets-T`, `Trends-T` (High Concept Drift).
    * **News Streams:** `News-T` (Stable Topics).
    * **Non-Temporal:** `SO-T` (Shuffled/No Temporal Dependency).

---

## 📂 Project Structure

```text
adaptive-textclust-validation/
├── datasets/                  # Directory for JSON-L dataset files
│   ├── Tweets-T               # (File without extension or .json)
│   ├── News-T
│   └── ...
├── run_benchmark.py           # Main execution script (Iterates all datasets)
├── run_experiment.py          # Single-dataset testing script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
````

## 🚀 Installation & Setup

Stream clustering libraries require low-level compilation for high performance. Please follow these steps carefully to avoid build errors.

### 1\. Prerequisites

  * **Python 3.10+**
  * **Microsoft C++ Build Tools** (Required for compiling River on Windows)
      * *Install via [Visual Studio Installer](https://visualstudio.microsoft.com/visual-cpp-build-tools/)*.
      * *Ensure "Desktop development with C++" workload is selected.*
  * **Rust Compiler** (Required for River's statistical modules)
      * *Install via [rustup.rs](https://rustup.rs/)*.

### 2\. Setup Environment

```bash
# Clone the repository
git clone [https://github.com/gautam9199/adaptive-textclust-validation.git](https://github.com/gautam9199/adaptive-textclust-validation.git)
cd adaptive-textclust-validation

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate

# Upgrade pip (Crucial for finding binary wheels)
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install river scikit-learn matplotlib pandas
```

-----

## 📊 Usage

### Run the Full Benchmark

To run the comparative analysis across all configured datasets with paper-specific parameters:

```bash
python experiment.py
```

**Output:**

  * The script will generate a specific `.png` plot for each dataset (e.g., `result_Tweets-T.png`).
  * Console logs will show NMI scores at every evaluation horizon (e.g., every 1,000 tweets).

### Configuration

The benchmark script uses the exact parameters specified in **Assenmacher & Trautmann (2022)** to ensure reproducibility. You can view these in `run_benchmark.py`. For example, `fading_factor` is set to `0.01` for Twitter datasets to handle fast concept drift.

```python
DATASET_CONFIG = {
    "Tweets-T": {
        "fading_factor": 0.01,  # High fading for fast social streams
        "tgap": 200,
        "horizon": 1000
    },
    "News-T": {
        "fading_factor": 0.001, # Low fading for stable news streams
        "tgap": 200,
        "horizon": 1000
    },
    # ...
}
```

-----

## 📈 Key Results

Our experiments confirm that the **Adaptive Threshold mechanism** significantly outperforms the **Fixed Threshold** approach on high-velocity social media streams.

  * **Stability:** The adaptive model maintains high NMI (\>0.85) even during sudden topic shifts on `Tweets-T` and `Trends-T`.
  * **Efficiency:** Prevents micro-cluster fragmentation compared to the fixed baseline.
  * **Conclusion:** The adaptive engine is essential for the "Suspicious Trend Filtering" phase of the detection framework.

-----

## 🎓 Credits & References

This work relies heavily on the theoretical foundations and algorithmic designs provided by the original authors.

### Primary Literature

  * **Assenmacher, D., & Trautmann, H. (2020).** *A Two-Phase Framework for Detecting Manipulation Campaigns in Social Media.* In HCII 2020.
  * **Assenmacher, D., & Trautmann, H. (2022).** *Textual One-Pass Stream Clustering with Automated Distance Threshold Adaptation.* In ACIIDS 2022.

### Tools & Libraries

  * **River:** Montiel, J., et al. *"River: machine learning for streaming data in Python."* (2021). [Website](https://riverml.xyz).

### Dataset Origins

The benchmark datasets (`Tweets-T`, `News-T`) are derived from standard short-text stream clustering research (Yin et al., 2018; Rakib et al., 2021) and were utilized in the original `textClust` evaluation studies.

-----

## 👤 Author

**Gautam Parmar**

  * Master's Student, Computer Science
  * University of Paderborn
  * Seminar: Textual Data Streams and Social Media Analytics (Winter 2025)
