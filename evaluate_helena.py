import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
import warnings
from collections import Counter

from data_prep_helena import get_helena_data
from models import SimpleMLP
from train_rebalance import train_rebalanced_model, train_standard_model, train_class_balanced_model

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

warnings.filterwarnings('ignore')

def get_shot_metrics(y_train, y_test, y_pred):
    train_counts = Counter(y_train)
    # Helena Shot Split
    many_shot = [c for c, count in train_counts.items() if count > 500]
    med_shot  = [c for c, count in train_counts.items() if 50 <= count <= 500]
    few_shot  = [c for c, count in train_counts.items() if count < 50]
    
    def acc_on_classes(classes):
        if not classes: return 0.0
        mask = np.isin(y_test, classes)
        if not np.any(mask): return 0.0
        return accuracy_score(y_test[mask], y_pred[mask])

    return {
        "Many-Shot Acc": acc_on_classes(many_shot),
        "Med-Shot Acc":  acc_on_classes(med_shot),
        "Few-Shot Acc":  acc_on_classes(few_shot)
    }

def save_figures(dataset_name, y_train, y_test, y_base, y_rebal, y_cb, tracking):
    slug = dataset_name.lower().replace("-", "_")
    classes, counts = np.unique(y_train, return_counts=True)
    
    # 1. Training Distribution
    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts, color='steelblue')
    plt.title(f"Class Distribution ({dataset_name})")
    plt.savefig(f"figure_1_{slug}_dist.png")
    plt.close()

    # 2. Unlearning Frequency
    plt.figure(figsize=(10, 5))
    unlearn_freqs = [np.mean(tracking['unlearning_freq'][y_train == c]) for c in np.unique(y_train)]
    plt.bar(np.unique(y_train), unlearn_freqs, color='#F97316')
    plt.title(f"Unlearning Frequency per Class")
    plt.savefig(f"figure_3_{slug}_unlearn.png")
    plt.close()

    # 3. Dynamics
    epochs_range = np.arange(tracking['loss_history'].shape[1])
    sorted_idx = np.argsort(tracking['unlearning_freq'])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, tracking['loss_history'][sorted_idx[0]], label='Easy')
    plt.plot(epochs_range, tracking['loss_history'][sorted_idx[-1]], label='Hard')
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, tracking['difficulty_history'][sorted_idx[0]], label='Easy')
    plt.plot(epochs_range, tracking['difficulty_history'][sorted_idx[-1]], label='Hard')
    plt.title("Difficulty vs Epochs")
    plt.legend()
    plt.savefig(f"figure_4_{slug}_dynamics.png")
    plt.close()

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def evaluate_helena():
    X_train, X_test, y_train, y_test, num_classes = get_helena_data()
    input_dim = X_train.shape[1]
    epochs = 100 
    batch_size = 256
    
    results = []
    
    print("Training Classical Baselines...", flush=True)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append({"Model": "Random Forest", "Overall Acc": accuracy_score(y_test, y_pred_rf), **get_shot_metrics(y_train, y_test, y_pred_rf)})
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results.append({"Model": "Logistic Regression", "Overall Acc": accuracy_score(y_test, y_pred_lr), **get_shot_metrics(y_train, y_test, y_pred_lr)})

    svm = LinearSVC(random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    results.append({"Model": "SVM (Linear)", "Overall Acc": accuracy_score(y_test, y_pred_svm), **get_shot_metrics(y_train, y_test, y_pred_svm)})

    print("Training Baseline MLP...", flush=True)
    set_seed(42)
    base_model = SimpleMLP(input_dim, num_classes, hidden_dim=256)
    base_model = train_standard_model(base_model, X_train, y_train, num_epochs=epochs, batch_size=batch_size)
    device = next(base_model.parameters()).device
    base_model.eval()
    with torch.no_grad():
        y_pred_base = base_model(torch.FloatTensor(X_test).to(device)).argmax(dim=1).cpu().numpy()
    results.append({"Model": "Baseline MLP", "Overall Acc": accuracy_score(y_test, y_pred_base), **get_shot_metrics(y_train, y_test, y_pred_base)})

    print("Training Class-Balanced MLP...", flush=True)
    set_seed(42)
    cb_model = SimpleMLP(input_dim, num_classes, hidden_dim=256)
    cb_model = train_class_balanced_model(cb_model, X_train, y_train, num_epochs=epochs, batch_size=batch_size)
    cb_model.eval()
    with torch.no_grad():
        y_pred_cb = cb_model(torch.FloatTensor(X_test).to(device)).argmax(dim=1).cpu().numpy()
    results.append({"Model": "Class-Balanced MLP", "Overall Acc": accuracy_score(y_test, y_pred_cb), **get_shot_metrics(y_train, y_test, y_pred_cb)})

    print("Training Re-balanced MLP (Proposed)...", flush=True)
    set_seed(42)
    rebal_model = SimpleMLP(input_dim, num_classes, hidden_dim=256)
    rebal_model, tracking = train_rebalanced_model(rebal_model, X_train, y_train, num_epochs=epochs, batch_size=batch_size, c=0.01)
    rebal_model.eval()
    with torch.no_grad():
        y_pred_rebal = rebal_model(torch.FloatTensor(X_test).to(device)).argmax(dim=1).cpu().numpy()
    results.append({"Model": "Re-balanced MLP", "Overall Acc": accuracy_score(y_test, y_pred_rebal), **get_shot_metrics(y_train, y_test, y_pred_rebal)})

    # Export Final Table
    df = pd.DataFrame(results)
    print("\n--- Final Helena Results ---")
    print(df)
    df.to_csv("table_helena_comparison.csv", index=False)
    
    # Export Figures
    save_figures("Helena", y_train, y_test, y_pred_base, y_pred_rebal, y_pred_cb, tracking)
    print("All exports completed successfully!", flush=True)

if __name__ == "__main__":
    evaluate_helena()
