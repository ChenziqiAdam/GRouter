from sentence_transformers import SentenceTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('/workspace/juhao/adaptive_agent/GDesigner/data/all-MiniLM-L6-v2')
model.to(device)
def get_sentence_embedding(sentence):
    embeddings = model.encode(sentence)
    return embeddings
