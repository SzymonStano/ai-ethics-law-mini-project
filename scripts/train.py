import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from src.config import PROJECT_ROOT as ROOT_DIR
from src.models import Eeg_GNN
from src.utils import load_data_for_subjects, get_weighted_loader, FocalLoss



BATCH_SIZE = 64
LR = 1.6e-5
WEIGHT_DECAY = 5.4e-4


EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EARLY_STOPPING_PATIENCE = 5
SAVE_MODEL_NAME = "best_model.pth"
SAVE_MODEL_PATH = ROOT_DIR / "models" / SAVE_MODEL_NAME

LABEL_INTERICTAL = 0
LABEL_ICTAL = 1
LABEL_PREICTAL = 2


SUBJECTS_TRAIN = [f"chb{str(i).zfill(2)}" for i in range(1, 21)]   
SUBJECTS_VAL = ['chb21', 'chb22']
SUBJECTS_TEST = ['chb23', 'chb24']


print(f"Trening na: {SUBJECTS_TRAIN}")
print(f"Walidacja na: {SUBJECTS_VAL}")
print(f"Test na:    {SUBJECTS_TEST}")


def run_training():
    train_data = load_data_for_subjects(SUBJECTS_TRAIN)
    val_data = load_data_for_subjects(SUBJECTS_VAL)
    test_data = load_data_for_subjects(SUBJECTS_TEST)

    # Skalowanie cech
    interictal_feats = torch.cat([g.x for g in train_data if g.y != LABEL_ICTAL], dim=0)
    if interictal_feats.shape[0] > 0:
        # Robust Scaler logic, fit na nienapadowych, transform na wszystkich
        median = torch.median(interictal_feats, dim=0).values
        iqr = torch.quantile(interictal_feats, 0.75, dim=0) - torch.quantile(interictal_feats, 0.25, dim=0) + 1e-6
        
        for g in train_data:
            g.x = (g.x - median) / iqr

        for g1 in val_data:
            g1.x = (g1.x - median) / iqr

        for g2 in test_data:
            g2.x = (g2.x - median) / iqr


    train_loader = get_weighted_loader(train_data, BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


    num_features = train_data[0].num_node_features
    
    model = Eeg_GNN(num_node_features=num_features, num_classes=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(gamma=2.0, alpha=None)

    best_val_auroc = 0 # dla early stopping
    history = {'loss': [], 'val_f1': [], 'val_auroc': []}

    print("\nRozpoczynam trening...")
    counts = 0
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Walidacja co epokę
        model.eval()
        y_true_v, y_pred_v = [], []
        y_true_v, y_prob_v = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)

                out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                y_pred_v.extend(out.argmax(dim=1).cpu().numpy())
                probs = F.softmax(out, dim=1).cpu().numpy()
                y_prob_v.append(probs.reshape(-1, probs.shape[-1]))
                y_true_v.append(batch.y.cpu().numpy().reshape(-1))
            
        y_true_v = np.concatenate(y_true_v)
        y_prob_v = np.concatenate(y_prob_v)

        # Binary ground truth: 1 = Pre or Ictal
        y_true_det = (y_true_v != LABEL_INTERICTAL).astype(int)

        y_true_det = y_true_v 
        y_score_det = y_prob_v[:, 1]
        # AUROC dla Non-Ictal vs Ictal
        val_det_auroc = roc_auc_score(y_true_det, y_score_det)


        val_f1 = f1_score(y_true_v, y_pred_v, average='macro')
        history['loss'].append(total_loss/len(train_loader))
        history['val_f1'].append(val_f1)
        history['val_auroc'].append(val_det_auroc)


        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Val Detection AUROC: {val_det_auroc:.4f}  | Val F1: {val_f1:.4f}" )

        if val_det_auroc > best_val_auroc:
            best_val_auroc = val_det_auroc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            counts = 0
        else:
            counts += 1
            if counts >= EARLY_STOPPING_PATIENCE:
                print(f"Brak poprawy przez {EARLY_STOPPING_PATIENCE} epok, przerywam trening.")
                break

    print("\n" + "="*30 + "\nWYNIKI NA PACJENTACH TESTOWYCH\n" + "="*30)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    model.eval()

    y_prob_t = []
    y_true_t, y_pred_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)

            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            y_pred_t.extend(out.argmax(dim=1).cpu().numpy())
            y_true_t.extend(batch.y.cpu().numpy().reshape(-1))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            y_prob_t.extend(probs)


    print(classification_report(y_true_t, y_pred_t, labels=[0, 1], 
                                target_names=['Non-Ictal', 'Ictal'], zero_division=0))

    y_true_t = np.array(y_true_t)
    y_prob_t = np.array(y_prob_t)

    # Obliczanie testowego AUROC
    y_true_det = y_true_t
    y_score_det = y_prob_t[:, 1] # Prawdopodobieństwo klasy Ictal

    test_det_auroc = roc_auc_score(y_true_det, y_score_det)
    print(f"\nDetection AUROC (Non-Ictal vs Ictal): {test_det_auroc:.4f}")


if __name__ == "__main__":
    run_training()