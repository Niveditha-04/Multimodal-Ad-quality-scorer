"""The classifier head's architecture and label set, shared between
train_classifier.py and the /score API. Kept separate so the API constructs
the exact same nn.Module shape used at training time before loading
classifier_head.pt -- redefining this independently in API code would risk
a silent architecture mismatch (e.g. a different hidden_dim) that
load_state_dict might not even catch cleanly.
"""
import torch.nn as nn

LABELS = ["approved", "policy_violation", "misleading", "low_quality"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

HIDDEN_DIM = 128
DROPOUT = 0.3


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)
