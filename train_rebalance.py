import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn.functional as F
import copy

def calculate_difficulty(p_curr, p_prev, y, c=1.0):
    """
    Implements the instance difficulty calculation from Yu et al.
    """
    N, K = p_curr.shape
    
    eps = 1e-10
    p_curr = np.clip(p_curr, eps, 1.0)
    p_prev = np.clip(p_prev, eps, 1.0)
    
    diff = p_curr - p_prev
    log_ratio = np.log(p_curr / p_prev)
    
    # Create mask for true labels
    y_mask = np.zeros((N, K), dtype=bool)
    y_mask[np.arange(N), y] = True
    
    # Unlearning (du)
    du_y = np.minimum(diff[y_mask], 0) * log_ratio[y_mask]
    
    du_other_matrix = np.maximum(diff, 0) * log_ratio
    du_other_matrix[y_mask] = 0 # Exclude y_i
    du_other = np.sum(du_other_matrix, axis=1)
    
    du = du_y + du_other
    
    # Learning (dl)
    dl_y = np.maximum(diff[y_mask], 0) * log_ratio[y_mask]
    
    dl_other_matrix = np.minimum(diff, 0) * log_ratio
    dl_other_matrix[y_mask] = 0 # Exclude y_i
    dl_other = np.sum(dl_other_matrix, axis=1)
    
    dl = dl_y + dl_other
        
    return du, dl

def train_rebalanced_model(model, X_train, y_train, num_epochs=100, batch_size=128, lr=1e-3, c=1.0, use_cumulative=True):
    """
    Trains the PyTorch model and tracks metrics for plotting.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    N = len(X_train)
    K = len(np.unique(y_train))
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    full_dataset = TensorDataset(X_tensor, y_tensor)
    
    w = np.ones(N) / N
    p_prev = np.ones((N, K)) / K
    
    sum_du = np.zeros(N)
    sum_dl = np.zeros(N)
    
    # Tracking metrics
    unlearning_counts = np.zeros(N)
    loss_history = np.zeros((N, num_epochs))
    difficulty_history = np.zeros((N, num_epochs))
    
    for epoch in range(num_epochs):
        print(f"  > Re-balanced Training: Epoch {epoch+1}/{num_epochs}", flush=True)
        sampler = WeightedRandomSampler(weights=w, num_samples=N, replacement=True)
        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=sampler)
        
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            p_curr = F.softmax(outputs, dim=1).cpu().numpy()
            
        # 3. Track Unlearning Frequency and Loss (Vectorized)
        # Check where probability decreased: p_curr[y] < p_prev[y]
        correct_p_curr = p_curr[np.arange(N), y_train]
        correct_p_prev = p_prev[np.arange(N), y_train]
        
        # Increment unlearning counts where probability dropped
        unlearning_counts += (correct_p_curr < (correct_p_prev - 1e-5)).astype(float)
        
        # Track Loss (Cross-Entropy per instance)
        loss_history[:, epoch] = -np.log(np.maximum(correct_p_curr, 1e-10))
            
        du, dl = calculate_difficulty(p_curr, p_prev, y_train, c=c)
        
        if use_cumulative:
            sum_du += du
            sum_dl += dl
            D = (c + sum_du) / (c + sum_dl)
        else:
            D = (c + du) / (c + dl)
            
        w = D / np.sum(D)
        
        # Track difficulty
        difficulty_history[:, epoch] = D
        
        p_prev = p_curr
        
    # Debugging: Print average weight per class
    print("\n[DEBUG] Average Sampling Weights per Class:")
    for c_idx in np.unique(y_train):
        mask = (y_train == c_idx)
        avg_w = np.mean(w[mask])
        count = np.sum(mask)
        print(f"  Class {c_idx}: count={count}, avg_weight={avg_w:.8f} (Rel to Uniform: {avg_w / (1/N):.2f}x)")

    unlearning_freq = (unlearning_counts / num_epochs) * 100.0
    
    print("\n[DEBUG] Average Unlearning Frequency per Class:")
    for c_idx in np.unique(y_train):
        mask = (y_train == c_idx)
        avg_freq = np.mean(unlearning_freq[mask])
        print(f"  Class {c_idx}: {avg_freq:.2f}%")
    
    metrics = {
        'unlearning_freq': unlearning_freq,
        'loss_history': loss_history,
        'difficulty_history': difficulty_history
    }
        
    return model, metrics

def train_standard_model(model, X_train, y_train, num_epochs=100, batch_size=128, lr=1e-3):
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        print(f"  > Processing Epoch {epoch+1}/{num_epochs}", flush=True)
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

def train_class_balanced_model(model, X_train, y_train, num_epochs=100, batch_size=128, lr=1e-3):
    """
    Trains a model using standard class-level balancing (inverse frequency).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    
    # Calculate class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    # Weights proportional to 1/frequency
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    # Assign weight to each sample based on its class
    sample_weights = torch.FloatTensor([class_weights[y] for y in y_train])
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_train), replacement=True)
    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, sampler=sampler)
    
    for epoch in range(num_epochs):
        print(f"  > Processing Epoch {epoch+1}/{num_epochs}", flush=True)
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    return model
