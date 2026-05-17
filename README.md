# Instance-Level Re-Balancing for Class-Imbalanced Classification

This repository implements a dynamic instance-level re-balancing strategy for training neural networks on class-imbalanced datasets, based on the research paper: *"A Re-Balancing Strategy for Class-Imbalanced Classification Based on Instance Difficulty"* (Yu et al., 2021).

## 🚀 Overview

Unlike traditional methods that re-balance data at the **class level** (e.g., Focal Loss, Class-Balanced Loss), this approach focuses on **instance difficulty**. It tracks the "learning speed" of every individual sample during training. Instances that the model frequently "unlearns" (where prediction probability decreases between epochs) are identified as harder and assigned higher sampling probabilities in subsequent epochs.

### Key Features
- **Dynamic Resampling:** Automatically adjusts sampling weights for each training instance after every epoch.
- **Unlearning Trend Tracking:** Measures how much the model "forgets" specific samples.
- **Support for Multiple Datasets:** Includes benchmarks for MIT-BIH (Arrhythmia) and HELENA datasets.
- **Comparative Analysis:** Compares the proposed method against standard Baseline and Class-Level Balancing (Inverse Frequency).

---

## 🧠 Core Concept: Instance Difficulty

The core of the strategy is the calculation of **Instance Difficulty ($D_{i,T}$)**, which is the ratio of accumulated **Unlearning Trend ($du$)** to accumulated **Learning Trend ($dl$)**.

1.  **Unlearning ($du$):** Captures worsening predictions for the ground-truth class and rising predictions for incorrect classes.
2.  **Learning ($dl$):** Captures improvement in predictions for the ground-truth class.
3.  **Sampling Weight ($w$):** Calculated by normalizing the difficulty $D$:
    $$w_{i,t} = \frac{D_{i,t}}{\sum_{j=1}^N D_{j,t}}$$

---

## 🛠️ Project Structure

- `models.py`: PyTorch implementations of `SimpleMLP` and `ECGCNN`.
- `train_rebalance.py`: Core logic for the re-balancing training loop and difficulty calculations.
- `data_prep_mitbih.py` / `data_prep_helena.py`: Data loading and preprocessing scripts.
- `evaluate_mitbih.py` / `evaluate_helena.py`: Main execution scripts that train models and generate result tables/figures.
- `rebalancing_failure_analysis.md`: Detailed report on experimental hurdles and edge cases.
- `mitbih/`: Contains the MIT-BIH Arrhythmia dataset CSV files.

---

## ⚙️ Installation

1. Clone the repository.
2. Ensure you have the following dependencies installed:
   ```bash
   pip install torch pandas numpy scikit-learn matplotlib
   ```
3. (Optional) Download additional datasets if required (MIT-BIH is already included in `mitbih/`).

---

## 📊 Usage

To run the full evaluation pipeline for the MIT-BIH dataset:

```bash
python evaluate_mitbih.py
```

This will:
1. Load the MIT-BIH data.
2. Train a **Baseline CNN**, a **Class-Balanced CNN**, and the **Proposed Re-balanced CNN**.
3. Generate accuracy metrics (Overall, Many-Shot, Med-Shot, Few-Shot).
4. Save comparison results to `table_mitbih_comparison.csv`.
5. Generate visualization plots:
   - `figure_1_mit_bih_dist.png`: Class distribution.
   - `figure_3_mit_bih_unlearn.png`: Unlearning frequency per class.
   - `figure_4_mit_bih_dynamics.png`: Training dynamics (Loss/Difficulty vs. Epochs).

---

## 📈 Results & Analysis

### Performance Metrics
The results are categorized into "Shots" based on class frequency in the training set:
- **Many-Shot:** Majority classes (>1000 samples).
- **Med-Shot:** Intermediate classes (100-1000 samples).
- **Few-Shot:** Minority classes (<100 samples).

### Experimental Findings
As detailed in `rebalancing_failure_analysis.md`:
- **Bias vs. Information:** The method effectively fixes model bias but cannot synthesize information for extremely low-sample classes (Few-Shot representation wall).
- **Stability:** The cumulative difficulty metric requires a sufficient number of epochs to stabilize.
- **Dataset Suitability:** Works best on datasets where "instance difficulty" is non-trivial (e.g., medical signals) rather than perfectly separable data.

---

## 📄 References
- Yu, et al. (2021). *A Re-Balancing Strategy for Class-Imbalanced Classification Based on Instance Difficulty.*
