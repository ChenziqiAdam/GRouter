import torch
import torch.nn.functional as F
from GDesigner.gnn.gcn import MLP
from typing import List

class Router():
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        self.mlp = MLP(input_size, hidden_size, output_size)

    def run(self, hidden_states: List[torch.Tensor], model_features: List[torch.Tensor]):
        embedding_list = [torch.cat([hidden_states[i][0], model_features[i]], dim=-1) for i in range(len(hidden_states))]
        embeddings = torch.stack(embedding_list, dim=0)
        logits = self.mlp(embeddings)
        return F.log_softmax(logits, dim=-1)
        