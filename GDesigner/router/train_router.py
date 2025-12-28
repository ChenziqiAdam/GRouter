import torch
import torch.nn.functional as F
import sys
import argparse
sys.path.append("/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner")
from GDesigner.gnn.gcn import MLP
from GDesigner.graph.graph import Graph
from GDesigner.llm.profile_embedding import get_sentence_embedding
import numpy as np
from typing import List, Any, Dict, Union, Literal
import random
import os
import json
import matplotlib.pyplot as plt

# class Router():
#     def __init__(self, input_size:int, hidden_size:int, output_size:int):
#         self.mlp = MLP(input_size, hidden_size, output_size)

#     def run(self, hidden_states: List[torch.Tensor], model_features: List[torch.Tensor]):
#         embedding_list = [torch.cat([hidden_states[i], model_features[i]], dim=-1) for i in range(len(hidden_states))]
#         embeddings = torch.stack(embedding_list, dim=0)
#         logits = self.mlp(embeddings)
#         return logits

def plot_train_log(jsonl_path: str, save_dir: str = "router_models"):
    steps, losses, accs = [], [], []

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 兼容你的字段名
            if "steps" not in r:
                continue

            steps.append(float(r["steps"]))
            if "loss" in r:
                losses.append(float(r["loss"]))
            else:
                losses.append(float("nan"))

            if "accuracy" in r:
                accs.append(float(r["accuracy"]))
            else:
                accs.append(float("nan"))

    # 保险：按 steps 排序（避免乱序写入）
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    steps  = [steps[i] for i in order]
    losses = [losses[i] for i in order]
    accs   = [accs[i] for i in order]

    # Loss 图
    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Loss vs Step")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"train_loss.png"), dpi=200)
    plt.close()

    # Accuracy 图
    plt.figure()
    plt.plot(steps, accs)
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs Step")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"train_acc.png"), dpi=200)
    plt.close()

def build_labels(success, cost, K=2.0, lam=1.0, beta=5.0, eps=1e-6):
    """
    success: (B, M) in {0,1}
    cost:    (B, M) raw cost for each model per task
    """
    B, M = success.shape

    # 1. across-model min-max normalization per task
    min_c = cost.min(dim=-1, keepdim=True).values
    max_c = cost.max(dim=-1, keepdim=True).values
    cost_norm = (cost - min_c) / (max_c - min_c + eps)  # (B, M)

    # 2. utility = success * (K - lam * cost_norm)
    utility = success * (K - lam * cost_norm)  # (B, M)

    # 3. softmax over utility
    logits_u = beta * utility
    q = torch.softmax(logits_u, dim=-1)  # (B, M)

    return q  # soft label

def get_loss(logits:torch.Tensor, labels:torch.Tensor):
    log_p = F.log_softmax(logits, dim=-1)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    loss = criterion(log_p, labels)
    return loss

def save_checkpoint(
    epoch,
    model,
    projector,
    optimizer,
    loss,
    save_dir
):
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "projector_state_dict": projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    path = os.path.join(save_dir, f"epoch_{epoch}.pt")
    torch.save(checkpoint, path)

def train_router(model: torch.nn.Module, projector: torch.nn.Module, train_data, test_data, num_epochs: int, batch_size: int, optimizer: torch.optim.Optimizer, model_paths, model_features, graphs, save_dir: str, save_checkpoint_every_epoch: bool=True):
    N = len(train_data)
    per_epoch_steps = N / batch_size
    for epoch in range(num_epochs):
        model.train()
        projector.train()
        # 简单手写一个 mini-batch 训练（这里数据不多，就直接整批也行）
        permutation = torch.randperm(N).tolist()  # 打乱
        shuffled_train_data = [train_data[i] for i in permutation]
        # train_data = shuffled_train_data
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for i in range(0, N, batch_size):
            batch_data = shuffled_train_data[i:i+batch_size]
            batch_x, batch_success, batch_cost = build_batch_data(batch_data, model_paths, model_features, graphs, projector)
            batch_labels = build_labels(batch_success, batch_cost)

            # 前向计算
            logits = model(batch_x).squeeze(-1)         # [batch_size, num_classes]
            loss = get_loss(logits, batch_labels)
            accuracy = (logits.argmax(dim=-1) == batch_labels.argmax(dim=-1)).float().mean()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            with open(os.path.join(save_dir, "train_loss.jsonl"), "a") as f:
                f.write(json.dumps({"epoch": epoch, "batch": i/batch_size, "steps": i/batch_size+epoch*per_epoch_steps,  "loss": loss.item(), "accuracy": accuracy.item()}) + "\n")
            print(f"Batch [{int(i/batch_size)}/{int(N/batch_size-1)}] - Loss: {loss.item():.4f}")
            print(f"Batch [{int(i/batch_size)}/{int(N/batch_size-1)}] - Accuracy: {accuracy.item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss/(N/batch_size):.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Accuracy: {epoch_accuracy/(N/batch_size):.4f}")
        model.eval()
        projector.eval()
        with torch.no_grad():
            test_loss, test_accuracy = validate_router(model, test_data, model_paths, model_features, graphs, batch_size, projector)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Test Loss: {test_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Test Accuracy: {test_accuracy:.4f}")
        with open(os.path.join(save_dir, "test_loss.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, "loss": test_loss, "accuracy": test_accuracy}) + "\n")
        if save_checkpoint_every_epoch:
            save_checkpoint(epoch, model, projector, optimizer, test_loss, save_dir=save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(projector.state_dict(), os.path.join(save_dir, "projector.pt"))
    plot_train_log(os.path.join(save_dir, "train_loss.jsonl"), save_dir=save_dir)

def validate_router(model, test_data, model_paths, model_features, graphs, batch_size: int, projector):
    print("Validating router...")
    N = len(test_data)
    test_loss = 0.0
    test_accuracy = 0.0
    for i in range(0, N, batch_size):
        # indices = range(i, i+batch_size)
        batch_data = test_data[i:i+batch_size]
        batch_x, batch_success, batch_cost = build_batch_data(batch_data, model_paths, model_features, graphs, projector)
        batch_labels = build_labels(batch_success, batch_cost)

        logits = model(batch_x).squeeze(-1)
        loss = get_loss(logits, batch_labels)
        accuracy = (logits.argmax(dim=-1) == batch_labels.argmax(dim=-1)).float().mean()
        test_loss += loss.item()
        test_accuracy += accuracy.item()
        print(f"Batch [{int(i/batch_size)}/{int(N/batch_size-1)}] - Loss: {loss.item():.4f}")
        print(f"Batch [{int(i/batch_size)}/{int(N/batch_size-1)}] - Accuracy: {accuracy.item():.4f}")
    return test_loss / (N/batch_size), test_accuracy / (N/batch_size)
    
def build_model_feature(model_description):
    query_embedding = torch.tensor(np.array(get_sentence_embedding(model_description)), dtype=torch.float32)
    return query_embedding

def build_model_features(model_descriptions):
    features = []
    for model_description in model_descriptions:
        feature = build_model_feature(model_description)
        features.append(feature)
    return features

# raw_data = [
#     {
#         "query": "query",
#         "success": {
#             "model_name": 1
#         },
#         "cost": {
#             "model_nmae": 0.05
#         },
#         "prompt_len": {
#             "model_name": 100
#         },
#         "completion_len": {
#             "model_name": 100
#         }
#     }
# ]

# X = [] # List of (M, D)

# success_list = [] # List of (M, 1)
# cost_list = [] # List of (M, 1)

def build_batch_data(raw_data, model_paths, model_features, graphs, projector):
    # model_paths = {
    #     "gpt-4o-mini": "graph_dir...",
    #     "gpt-4o": "graph_dir...",
    # }
    # model_descriptions = {
    #     "gpt-4o-mini": "des",
    #     "gpt-4o": "des",
    # }
    X = []
    success_list = []
    cost_list = []
    for index in range(len(raw_data)):
        current_data = raw_data[index]
        query = current_data["query"]
        hidden_states = [graph.gvae_encoder(query)[0] for graph in graphs]
        embedding_list = [torch.cat([hidden_states[i], projector(model_features[i])], dim=-1) for i in range(len(hidden_states))]
        embeddings = torch.stack(embedding_list, dim=0)
        # embeddings = torch.stack(hidden_states, dim=0)
        X.append(embeddings)
        success_item = [current_data["success"][model] for model in model_paths]
        cost_item = [current_data["cost"][model] for model in model_paths]
        success_tensor = torch.tensor(success_item, dtype=torch.float32)
        cost_tensor = torch.tensor(cost_item, dtype=torch.float32)
        success_list.append(success_tensor)
        cost_list.append(cost_tensor)
    X = torch.stack(X, dim=0)
    success_list = torch.stack(success_list, dim=0)
    cost_list = torch.stack(cost_list, dim=0)
    return X, success_list, cost_list

def train_test_split(raw_data, train_ratio=0.84):
    random.shuffle(raw_data)
    train_data = raw_data[:int(len(raw_data) * train_ratio)]
    test_data = raw_data[int(len(raw_data) * train_ratio):]
    # train_X, train_success_list, train_cost_list = build_data(train_data, graph_kwargs, model_paths, model_descriptions)
    # test_X, test_success_list, test_cost_list = build_data(test_data, graph_kwargs, model_paths, model_descriptions)
    return train_data, test_data

def _get_raw_data(result_dir: str):
    _raw_data = {}
    result_file_path = os.path.join(result_dir, "result.json")
    result_file = json.load(open(result_file_path, "r"))
    for dir in os.listdir(result_dir):
        if not os.path.isdir(os.path.join(result_dir, dir)):
            continue
        group_conversation_history = json.load(open(os.path.join(result_dir, dir, "group_conversation_history.json"), "r"))
        current_task = group_conversation_history[0]["task"]
        current_task_total_cost = 0
        for conversation in group_conversation_history[1:]:
            current_task_total_cost += conversation["price"]
        index = int(dir)-1
        success = result_file[index]["Solved"]
        _raw_data[current_task] = {
            "success": success,
            "cost": current_task_total_cost,
        }
    return _raw_data

def update_raw_data(raw_data: List[Any], _raw_data: Dict[str, Any], model_name: str):
    if raw_data == []:
        for query in _raw_data:
            raw_data.append({
                "query": query,
                "success": {
                    model_name: _raw_data[query]["success"]
                },
                "cost": {
                    model_name: _raw_data[query]["cost"]
                }

            })
    else:
        for index in range(len(raw_data)):
            query = raw_data[index]["query"]
            raw_data[index]["success"][model_name] = _raw_data[query]["success"]
            raw_data[index]["cost"][model_name] = _raw_data[query]["cost"]
    return raw_data


def get_raw_data(model_result_dir_map: Dict[str, str]):
    raw_data = []
    for model_name, result_dir in model_result_dir_map.items():
        _raw_data = _get_raw_data(result_dir)
        raw_data = update_raw_data(raw_data, _raw_data, model_name)
    return raw_data

def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on gsm8k")
    parser.add_argument("--dataset_json", type=str, default="datasets/gsm8k/gsm8k.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=3e-4,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--num_epochs',type=int,default=3,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default=10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="gsm8k",help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the GDesigner')
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    parser.add_argument('--gumbel_tau', type=float, default=1,
                        help="Gumbel-Softmax temperature for edge sampling.")
    parser.add_argument('--refine_rank', type=int, default=2,
                        help="Rank used in the refine module (default uses full rank).")
    parser.add_argument('--refine_zeta', type=float, default=1e-1,
                        help="Nuclear norm regularization strength for the refine module.")
    parser.add_argument('--anchor_weight', type=float, default=1.0,
                        help="Weight of the anchor loss during training.")
    parser.add_argument('--sparse_weight', type=float, default=1.0,
                        help="Weight of the sparse regularization loss during training.")
    parser.add_argument('--from_graph_dir', type=str, default=None,
                        help="Directory to load the graph.")
    parser.add_argument('--to_graph_dir', type=str, default=None,
                        help="Directory to save the graph.")
    parser.add_argument('--experiment_name', type=str, default=None,
                        help="Name of the experiment.")
    parser.add_argument('--dataset_start_index', type=int, default=0,
                        help="Start index of the dataset.")
    parser.add_argument('--num_of_data', type=int, default=None,
                        help="Number of data to run.")
    parser.add_argument('--rebuild_raw_data', action='store_true',
                        help="Rebuild the raw data.")
    parser.add_argument('--raw_data_path', type=str, default="data/router_data/raw_data.json",
                        help="Path to the raw data.")
    parser.add_argument('--save_dir', type=str, default="router_models/gpt-4.1-nano_gpt-4o-mini_deepseek-v3_lr_3e-4",
                        help="Directory to save the model.")
    parser.add_argument('--save_checkpoint_every_epoch', action='store_true',
                        help="Save the model every epoch.")
    args = parser.parse_args()
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args

def main():
    args = parse_args()
    args.save_checkpoint_every_epoch = True
    # args.rebuild_raw_data = True
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    if args.rebuild_raw_data:
        # model_result_dir_map = {
        #     "gpt-4o-mini": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/result/gsm8k_gpt4o-mini/gsm8k_gpt-4o-mini",
        #     "qwen3-32b": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/result/gsm8k_qwen3-32b/gsm8k_qwen3-32b",
        #     "deepseek-v3": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/result/gsm8k_deepseek-v3/gsm8k_deepseek-v3",
        #     "gpt-4.1-nano": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/result/gsm8k_gpt-4.1-nano/gsm8k_gpt-4.1-nano",
        #     "gpt-5-nano": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/result/gsm8k_gpt-5-nano/gsm8k_gpt-5-nano",
        # }
        model_result_dir_map = {
            "gpt-4o-mini": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/result/gsm8k_gpt4o-mini/gsm8k_gpt-4o-mini",
            "deepseek-v3": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/result/gsm8k_deepseek-v3/gsm8k_deepseek-v3",
            "gpt-4.1-nano": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/result/gsm8k_gpt-4.1-nano/gsm8k_gpt-4.1-nano",
        }
        raw_data = get_raw_data(model_result_dir_map)
        with open(args.raw_data_path, "w") as file:
            json.dump(raw_data, file, indent=4)
    else:
        with open(args.raw_data_path, "r") as file:
            raw_data = json.load(file)
    # model_paths = {
    #     "gpt-4o-mini": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_gsm8k_gpt4o-mini",
    #     "qwen3-32b": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_gsm8k_qwen3-32b",
    #     "deepseek-v3": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_gsm8k_deepseek-v3",
    #     "gpt-4.1-nano": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_gsm8k_gpt4.1-nano",
    #     "gpt-5-nano": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_gsm8k_gpt-5-nano",
    # }
    # model_descriptions = {
    #     "gpt-4o-mini": "gpt-4o-mini is a small, cost-efficient version of GPT-4o that supports text and image inputs, offers low-latency responses with a large context window, and delivers strong performance on everyday reasoning, coding, and math tasks.",
    #     "qwen3-32b": "Qwen3-32B is a 32.8 billion parameter dense language model from the Qwen3 series, featuring hybrid thinking modes, strong complex reasoning and instruction-following abilities, multilingual support across 100+ languages, and long-context handling for advanced agent and tool-use scenarios.",
    #     "deepseek-v3": "DeepSeek-V3 is a large Mixture-of-Experts language model with 671 billion total parameters and 37 billion activated per token, combining Multi-head Latent Attention and efficient MoE architecture to provide high-quality reasoning and coding capabilities with fast, cost-effective inference over long contexts.",
    #     "gpt-4.1-nano": "gpt-4.1-nano is a compact and cost-efficient model in the GPT-4.1 family, designed for fast text generation and reliable instruction-following, providing solid text understanding and basic reasoning capabilities for everyday language processing tasks at scale.",
    #     "gpt-5-nano": "gpt-5-nano is a compact and highly cost-efficient model in the GPT-5 family, designed for fast text generation and reliable instruction-following, with improved reasoning quality and stability over earlier nano-scale models, making it suitable for everyday language understanding and generation tasks at scale."
    # }
    model_paths = {
        "gpt-4o-mini": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_gsm8k_gpt4o-mini",
        "deepseek-v3": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_gsm8k_deepseek-v3",
        "gpt-4.1-nano": "/Users/chenjunzhi/Desktop/context_efficient_agent/GDesigner/graph_result/train_gsm8k_gpt4.1-nano",
    }
    model_descriptions = {
        "gpt-4o-mini": "gpt-4o-mini is a small, cost-efficient version of GPT-4o that supports text and image inputs, offers low-latency responses with a large context window, and delivers strong performance on everyday reasoning, coding, and math tasks.",
        "deepseek-v3": "DeepSeek-V3 is a large Mixture-of-Experts language model with 671 billion total parameters and 37 billion activated per token, combining Multi-head Latent Attention and efficient MoE architecture to provide high-quality reasoning and coding capabilities with fast, cost-effective inference over long contexts.",
        "gpt-4.1-nano": "gpt-4.1-nano is a compact and cost-efficient model in the GPT-4.1 family, designed for fast text generation and reliable instruction-following, providing solid text understanding and basic reasoning capabilities for everyday language processing tasks at scale.",
    }
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode,len(agent_names))
    graph_kwargs = dict(
        optimized_spatial=True,
        optimized_temporal=False,
        **kwargs
    )
    train_data, test_data = train_test_split(raw_data)
    model_features = build_model_features(model_descriptions)
    graphs = [Graph.load_graph(model_paths[model], **graph_kwargs) for model in model_paths]
    for graph in graphs:
        graph.eval()
    router_model = MLP(32, 64, 1)
    projector = MLP(384, 64, 16)
    router_model.train()
    projector.train()
    optimizer = torch.optim.Adam(list(router_model.parameters()) + list(projector.parameters()), lr=args.lr)
    train_router(router_model, projector, train_data, test_data, args.num_epochs, args.batch_size, optimizer, model_paths, model_features, graphs, args.save_dir, args.save_checkpoint_every_epoch)


def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']]
               ,N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Programming Expert'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}   

if __name__ == "__main__":
    main()