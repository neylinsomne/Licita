import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

# --- DEFINICIÓN DEL MODELO (Para uso futuro) ---
class LicitacionGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.gnn = to_hetero(self.conv1, metadata, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.gnn(x_dict, edge_index_dict)
        return x_dict['licitacion']

# --- UTILIDAD ACTUAL (Mean Pooling) ---
# Usamos esto mientras no tengas el modelo entrenado
def generate_doc_vector_simple(embedder, chunks_data, title_taxonomy):
    """Genera un vector promedio del documento + taxonomía inferida"""
    # 1. Vector de la familia principal
    tax_vec = embedder.encode(title_taxonomy)
    # 2. Promedio de los primeros 5 chunks (los más relevantes)
    chunks_text = [c['text'] for c in chunks_data[:5]]
    if not chunks_text: return tax_vec.tolist()
    
    chunks_vec = embedder.encode(chunks_text)
    mean_vec = chunks_vec.mean(axis=0)
    
    # Promedio ponderado (50% taxonomia, 50% contenido)
    final_vec = (tax_vec + mean_vec) / 2
    return final_vec.tolist()