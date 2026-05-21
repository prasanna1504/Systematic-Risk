import os
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import sys

# Append parent dir so python can resolve modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rasr_ge import RASR_GE
from training.dataset import FinancialGraphDataset

def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(logits, batch.y)
            total_loss += loss.item() * batch.num_graphs

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_targets.append(batch.y.cpu())

    avg_loss = total_loss / len(loader.dataset)

    y_true = torch.cat(all_targets).numpy()
    y_prob = torch.cat(all_preds).numpy()
    # Use calibrated threshold from config (default 0.5 keeps early epochs stable)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5 # Default if only one class exists

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        'loss': avg_loss,
        'auroc': auroc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Hyperparams
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = float(config['training']['lr'])
    wd = float(config['training']['weight_decay'])
    patience = config['training']['early_stopping_patience']
    pos_w = config['training']['pos_weight']
    eval_threshold = float(config['training'].get('eval_threshold', 0.5))
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Datasets
    root_dir = os.path.abspath('.')
    print("Loading Train Dataset...")
    train_dataset = FinancialGraphDataset(root_dir, split='train', seq_len=config['model']['seq_len'])
    print("Loading Val Dataset...")
    val_dataset = FinancialGraphDataset(root_dir, split='val', seq_len=config['model']['seq_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = RASR_GE(
        seq_len=config['model']['seq_len'],
        input_dim=5,
        lstm_hidden=config['model']['lstm_hidden'],
        lstm_layers=config['model']['lstm_layers'],
        gat_heads=config['model']['gat_heads'],
        gat_out_dim=config['model']['gat_out_dim'],
        dropout=config['model']['dropout']
    ).to(device)
    
    pos_weight = torch.tensor([pos_w]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_auroc = 0.0
    stagnant_epochs = 0
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Starting Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        # tqdm for epochs
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(logits, batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = total_loss / len(train_dataset)
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=eval_threshold)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} "
              f"| Val AUROC: {val_metrics['auroc']:.4f} | F1: {val_metrics['f1']:.4f} "
              f"| Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}")
              
        scheduler.step(val_metrics['auroc'])
        
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            stagnant_epochs = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            print(f"--> Saved new best model (AUROC: {best_auroc:.4f})")
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                print("Early stopping triggered!")
                break

if __name__ == "__main__":
    train_model()
