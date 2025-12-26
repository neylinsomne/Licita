import torch
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T  # <--- AGREGA ESTA LÍNEA
# ==========================================
# 1. CONSTRUCTOR DE GRAFOS (Raw Data -> HeteroData)
# ==========================================
def build_graph_for_inference(chunks_data, embedder):
    """
    Convierte la lista de diccionarios de chunks en un objeto Grafo Heterogéneo.
    Se usa tanto para entrenar como para generar embeddings avanzados.
    """
    data = HeteroData()

    # --- A. PREPARAR NODOS ---
    
    # 1. Nodos Chunk
    # Extraemos el texto de cada chunk y lo vectorizamos
    chunk_texts = [c['text'] for c in chunks_data]
    chunk_embeddings = embedder.encode(chunk_texts, convert_to_tensor=True)
    data['chunk'].x = chunk_embeddings # Shape: [num_chunks, 768]

    # 2. Nodos Concepto (Entidades extraídas)
    # Recopilamos todos los conceptos únicos encontrados en este documento
    # Nota: Asumimos que 'entities' o 'concepts' viene en el dict del chunk. 
    # Si usas el parser simple, quizás debas inferirlos o usar palabras clave.
    all_concepts = []
    chunk_to_concept_edges = []
    
    # Mapeo temporal para saber el ID de cada concepto único
    concept_map = {} 
    
    for i, chunk in enumerate(chunks_data):
        # Si tu parser extrajo entidades (JSONB), úsalas. 
        # Si no, usamos palabras clave simples del título/categoría como "conceptos"
        # para enriquecer el grafo.
        simulated_concepts = [chunk['category']] # Por defecto usamos la categoría
        
        # Si tienes extracción real de IA, descomenta esto:
        # if 'extracted_concepts' in chunk: simulated_concepts.extend(chunk['extracted_concepts'])

        for concept in simulated_concepts:
            if concept not in concept_map:
                concept_map[concept] = len(all_concepts)
                all_concepts.append(concept)
            
            # Crear arista: Chunk[i] -> Concepto[ID]
            concept_id = concept_map[concept]
            chunk_to_concept_edges.append([i, concept_id])

    # Vectorizar Conceptos
    if all_concepts:
        concept_embeddings = embedder.encode(all_concepts, convert_to_tensor=True)
        data['concepto'].x = concept_embeddings
    else:
        # Fallback si no hay conceptos: vector de ceros
        data['concepto'].x = torch.zeros((1, 768))

    # 3. Nodo Licitación (Nodo Raíz)
    # Inicializamos con el promedio simple de chunks como punto de partida
    doc_embedding = torch.mean(chunk_embeddings, dim=0, keepdim=True)
    data['licitacion'].x = doc_embedding # Shape: [1, 768]

    # --- B. PREPARAR ARISTAS (Indices de Adyacencia) ---

    # Edge: Licitacion -> Contiene -> Chunk
    # El nodo 0 de licitación se conecta a todos los chunks (0 a N)
    num_chunks = len(chunks_data)
    edge_index_lic_chunk = torch.tensor([
        [0] * num_chunks,       # Source: Licitacion ID 0
        list(range(num_chunks)) # Target: Chunk IDs
    ], dtype=torch.long)
    data['licitacion', 'contiene', 'chunk'].edge_index = edge_index_lic_chunk

    # Edge: Chunk -> Menciona -> Concepto
    if chunk_to_concept_edges:
        edge_index_chunk_conc = torch.tensor(chunk_to_concept_edges, dtype=torch.long).t().contiguous()
        data['chunk', 'menciona', 'concepto'].edge_index = edge_index_chunk_conc
    else:
        data['chunk', 'menciona', 'concepto'].edge_index = torch.empty((2, 0), dtype=torch.long)

    data = T.ToUndirected()(data)  # <--- AGREGA ESTO
    
    return data

# ==========================================
# 2. DEFINICIÓN DEL MODELO (Arquitectura)
# ==========================================
class LicitacionGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        # Usamos SAGEConv porque es inductivo (sirve para nodos nuevos no vistos)
        # (-1, -1) permite que PyTorch infiera el tamaño de entrada automáticamente
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        
        # Transformamos la conv simple en una GNN Heterogénea
        # Esto crea una convolución única para cada tipo de relación
        self.gnn = to_hetero(self.conv1, metadata, aggr='mean')
        
        # Segunda capa (opcional, para mayor profundidad)
        self.gnn2 = to_hetero(self.conv2, metadata, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        # Paso 1: Mensaje pasando capa 1 + Activación ReLU
        x_dict = self.gnn(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        # Paso 2: Mensaje pasando capa 2 (Salida lineal)
        # Nota: Podrías usar self.gnn2 aquí si inicializas to_hetero con conv2
        # Por simplicidad en MVP, a veces solo una capa basta, pero definamos el flujo completo:
        # x_dict = self.gnn2(x_dict, edge_index_dict)
        
        # Retornamos el vector latente actualizado del nodo 'licitacion'
        # Este vector ahora contiene información condensada de todos sus chunks y conceptos
        return x_dict['licitacion']

# ==========================================
# 3. UTILIDADES DE GENERACIÓN (Bridge)
# ==========================================

def generate_doc_vector_simple(embedder, chunks_data, title_taxonomy):
    """
    MÉTODO BASE: Promedio Ponderado (Sin GNN entrenada).
    Usa esto mientras recolectas datos para entrenar la red.
    """
    # 1. Vector de la taxonomía (Peso alto porque es inferencia global)
    tax_vec = embedder.encode(title_taxonomy)
    
    # 2. Vector del contenido (Solo texto)
    # Filtramos chunks vacíos
    valid_chunks = [c['text'] for c in chunks_data if len(c['text']) > 10]
    
    if not valid_chunks:
        return tax_vec.tolist()
    
    chunks_vec = embedder.encode(valid_chunks[:10]) # Max 10 chunks para velocidad
    mean_vec = chunks_vec.mean(axis=0)
    
    # Fusión: 40% Taxonomía + 60% Contenido
    final_vec = (tax_vec * 0.4) + (mean_vec * 0.6)
    
    return final_vec.tolist()

def generate_doc_vector_advanced(embedder, chunks_data, model_path=None):
    """
    MÉTODO AVANZADO: Usa la GNN.
    Requiere que hayas entrenado y guardado un 'model.pth'.
    """
    # 1. Construir el grafo al vuelo
    data = build_graph_for_inference(chunks_data, embedder)
    
    # 2. Inicializar modelo (si no está cargado)
    # Nota: Necesitas saber hidden_channels usado en entrenamiento
    model = LicitacionGNN(hidden_channels=64, out_channels=768, metadata=data.metadata())
    
    if model_path:
        model.load_state_dict(torch.load(model_path))
        model.eval()
    
    with torch.no_grad():
        vector_final = model(data.x_dict, data.edge_index_dict)
        
    return vector_final.squeeze().tolist()