import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from src.config import PROCESSED_DIR
from torch_geometric.utils import add_self_loops

LABEL_ICTAL = 1


def load_data_for_subjects(subject_list, remap_to_binary=True, add_self_loops_flag=False, scaling=False):
    combined_data = []
    if add_self_loops_flag:
        print("Ładowanie danych z dodanymi pętlami własnymi!")
    if scaling:
        print("Ładowanie danych z normalizacją cech per pacjent!")
    for subj in subject_list:
        file_path = PROCESSED_DIR / f"patient_{subj}.pt"

        if file_path.exists():
            data = torch.load(file_path, weights_only=False)
            for d in data:
                d.x = d.x.float()
                d.edge_index = d.edge_index.long()

                if remap_to_binary:
                    d.y = (d.y == LABEL_ICTAL).long()

                if add_self_loops_flag:
                    d.edge_index, d.edge_attr = add_self_loops(
                        d.edge_index,
                        edge_attr=d.edge_attr,
                    fill_value=1.0 
                    )

            if scaling:
                interictal_feats = torch.cat([g.x for g in data if g.y != 1], dim=0)
                if interictal_feats.shape[0] > 0:
                    median = torch.median(interictal_feats, dim=0).values
                    iqr = torch.quantile(interictal_feats, 0.75, dim=0) - torch.quantile(interictal_feats, 0.25, dim=0) + 1e-6
                    
                    for g in data:
                        g.x = (g.x - median) / iqr
                            
            combined_data.extend(data)
        else:
            print(f"Brak pliku dla {subj}")
    return combined_data


def get_weighted_loader(data_list, batch_size):
    labels = [int(d.y.item()) for d in data_list]
    counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    # return DataLoader(data_list, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
    return DataLoader(data_list, batch_size=batch_size, sampler=sampler)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_p = F.log_softmax(inputs, dim=1)
        p = torch.exp(log_p)
        
        log_p_target = log_p.gather(1, targets.view(-1, 1)).view(-1)
        p_target = p.gather(1, targets.view(-1, 1)).view(-1)

        loss = -1 * (1 - p_target)**self.gamma * log_p_target

        if self.alpha is not None:
            alpha_weight = self.alpha.to(inputs.device).gather(0, targets)
            loss = alpha_weight * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

