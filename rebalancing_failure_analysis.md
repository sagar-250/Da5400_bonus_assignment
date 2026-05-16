# Experimental Failure Analysis: Instance-Level Re-balancing Strategy

This document outlines the technical hurdles, conceptual failures, and performance bottlenecks encountered during the implementation of the "Instance Difficulty" re-balancing strategy across various datasets.

---

## 1. Dataset-Specific Failures

### **A. NSL-KDD (Cybersecurity)**
*   **Failure Type:** "Saturation Failure"
*   **Observation:** Baseline models (Random Forest, MLP) achieved >99% accuracy almost immediately.
*   **Root Cause:** The dataset features were too linearly separable or the specific attack patterns were too distinct.
*   **Re-balancing Impact:** Neutral. Because the model "learned" everything within 1-2 epochs, there were zero "unlearning" events to track, making the dynamic sampling weights default to uniform.

### **B. Arrhythmia (Small-Scale Medical)**
*   **Failure Type:** "Generalization Failure"
*   **Observation:** Re-balanced MLP slightly improved Med-Shot accuracy but failed to beat the Baseline Overall.
*   **Root Cause:** 
    *   **Data Scarcity:** 452 samples was insufficient to provide a stable "learning speed" signal. Unlearning events were likely stochastic noise.
    *   **Dimensionality:** 279 features for <500 samples led to instant memorization (overfitting), preventing the dynamic weights from evolving.
*   **Lesson:** High-complexity instance-level weighting requires a minimum data volume to distinguish "difficulty" from "noise."


## 2. Technical Implementation Hurdles

### **A. Calculation of Prediction Variations (PSI)**
*   **Issue:** Early versions of the `calculate_difficulty` function were susceptible to numerical instability.
*   **Fix:** Implementation of `np.clip` (1e-10) was necessary to prevent `log(0)` or division by zero when the model became overconfident on "easy" instances.


### **C. The "Epoch Stability Lag"**
*   **Observation:** Training for few epochs (e.g., <20) often resulted in re-balanced models performing worse than the baseline.
*   **Root Cause:** The cumulative difficulty metric ($D_{i,T}$) is a ratio of accumulated trends. In early epochs, the denominator and numerator are dominated by the random initialization of the network.
*   **Lesson:** The strategy requires a sufficient observation window to allow learning/unlearning trends to stabilize. If the training is too short, the sampling weights act as "stochastic noise" rather than a corrective signal, potentially leading to diverging gradients or unstable class boundaries.

---

## 3. Algorithmic Edge Cases

### **A. The "Few-Shot Representation" Wall**
*   **Observation:** In the Arrhythmia dataset, classes with only 2 samples did not improve even with 2.0x sampling weights.
*   **Conclusion:** Instance-level re-balancing can fix **bias**, but it cannot synthesize **information**. If the minority samples are not representative, re-weighting them just forces the model to overfit on outliers.

### **B. Model-Data Mismatch**
*   **Observation:** For ECG signals (MIT-BIH), standard MLPs performed poorly regardless of balancing.
*   **Fix:** Shifted to **1D CNNs** to capture temporal patterns. Re-balancing works best when the model architecture is actually capable of learning the underlying features.
