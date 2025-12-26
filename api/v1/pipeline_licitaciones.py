import fitz  # PyMuPDF
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURACIÓN Y MODELOS ----

embedder = SentenceTransformer('all-MiniLM-L6-v2') 

def get_llm_extraction(chunk_text):
    """
    Simula la llamada a DeepSeek/Gemini.
    Aquí es donde le pides: "Extrae productos y sugiere UNSPSC".
    """
    if "cámara" in chunk_text.lower():
        return {"entidades": ["Cámara PTZ", "Video Vigilancia"], "unspsc_family": "46171600"}
    elif "limpieza" in chunk_text.lower():
        return {"entidades": ["Servicio de Aseo", "Insumos Cafetería"], "unspsc_family": "76111500"}
    return {"entidades": [], "unspsc_family": None}

# --- 2. PROCESAMIENTO DEL PDF ---
def process_pdf(pdf_path, doc_id="lic_001"):
    doc = fitz.open(pdf_path)
    chunks_data = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        raw_chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        
        for i, chunk_text in enumerate(raw_chunks):
            extraction = get_llm_extraction(chunk_text)
            
            if extraction["entidades"]: 
                chunk_emb = embedder.encode(chunk_text)
                
                chunks_data.append({
                    "chunk_id": f"{doc_id}_p{page_num}_c{i}",
                    "text": chunk_text,
                    "embedding": chunk_emb,
                    "entities": extraction["entidades"],
                    "unspsc": extraction["unspsc_family"]
                })
    return chunks_data

# --- 3. CONSTRUCCIÓN DEL GRAFO HETEROGÉNEO (GNN) ---
def build_hetero_graph(chunks_data):
    data = HeteroData()
    
    # -- NODOS --
    # Tipo A: Licitación (En este caso, un nodo agregado o "Root")
    # Representamos la licitación como el promedio de sus chunks (inicialmente)
    all_embeddings = [c["embedding"] for c in chunks_data]
    doc_embedding = torch.tensor(all_embeddings).mean(dim=0).unsqueeze(0)
    data['licitacion'].x = doc_embedding # Feature matrix [1, 384]

    # Tipo B: Chunks
    chunk_embeddings = torch.tensor([c["embedding"] for c in chunks_data])
    data['chunk'].x = chunk_embeddings # Feature matrix [N_chunks, 384]

    # Tipo C: Conceptos (Entidades extraídas)
    # Primero, unificar vocabulario de conceptos únicos
    unique_concepts = list(set([e for c in chunks_data for e in c['entities']]))
    concept_map = {concept: i for i, concept in enumerate(unique_concepts)}
    
    # Vectorizamos los conceptos (Node Features)
    if unique_concepts:
        concept_embeddings = embedder.encode(unique_concepts, convert_to_tensor=True)
        data['concepto'].x = concept_embeddings
    else:
         # Fallback si no hay conceptos
        data['concepto'].x = torch.zeros((1, 768))

    # -- ARISTAS (EDGES) --
    # Edge 1: Licitacion -> Tiene -> Chunk
    # (El nodo 0 de licitación se conecta a todos los chunks)
    edge_index_doc_chunk = torch.tensor([
        [0] * len(chunks_data), # Origen (Licitacion ID 0)
        list(range(len(chunks_data))) # Destino (Chunk IDs 0 to N)
    ], dtype=torch.long)
    data['licitacion', 'contiene', 'chunk'].edge_index = edge_index_doc_chunk

    # Edge 2: Chunk -> Menciona -> Concepto
    src_chunk = []
    dst_concept = []
    
    for chunk_idx, item in enumerate(chunks_data):
        for ent in item['entities']:
            if ent in concept_map:
                src_chunk.append(chunk_idx)
                dst_concept.append(concept_map[ent])
    
    if src_chunk:
        edge_index_chunk_concept = torch.tensor([src_chunk, dst_concept], dtype=torch.long)
        data['chunk', 'menciona', 'concepto'].edge_index = edge_index_chunk_concept
    else:
        data['chunk', 'menciona', 'concepto'].edge_index = torch.empty((2, 0), dtype=torch.long)

    return data

# --- 4. MODELO GNN Y CONTRASTIVE LOSS ---

class LicitacionGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        # Usamos GraphSAGE para procesar grafos heterogéneos
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.gnn = to_hetero(self.conv1, metadata, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.gnn(x_dict, edge_index_dict)
        return x_dict['licitacion'] # Retornamos el vector latente de la licitación

# Función de pérdida contrastiva (Triplet Loss simplificada)
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        label: 1 si son similares (misma categoría), 0 si son diferentes
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# --- 5. EJECUCIÓN PRINCIPAL ---

if __name__ == "__main__":
    # A. Crear datos dummy (simulando lectura PDF)
    print("1. Procesando PDF simulado...")
    # En tu caso real: process_pdf("ruta/a/tu/archivo.pdf")
    # Simulamos datos de 2 licitaciones:
    # Lic 1: Cámaras (Seguridad)
    chunks_lic_A = process_pdf("dummy.pdf", doc_id="lic_camaras") 
    # Lic 2: Insumos Aseo (Diferente)
    chunks_lic_B = process_pdf("dummy.pdf", doc_id="lic_aseo")
    # Lic 3: Video Vigilancia (Similar a A)
    chunks_lic_C = process_pdf("dummy.pdf", doc_id="lic_vigilancia") 


    print("2. Construyendo Grafos Heterogéneos...")
    graph_A = build_hetero_graph(chunks_lic_A)
    graph_B = build_hetero_graph(chunks_lic_B)
    graph_C = build_hetero_graph(chunks_lic_C)

   
    model = LicitacionGNN(hidden_channels=64, out_channels=32, metadata=graph_A.metadata())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = ContrastiveLoss()

    
    print("3. Entrenando con Contrastive Loss...")
    model.train()

    vec_A = model(graph_A.x_dict, graph_A.edge_index_dict)
    vec_B = model(graph_B.x_dict, graph_B.edge_index_dict)
    vec_C = model(graph_C.x_dict, graph_C.edge_index_dict)

    # Calculamos Loss
    # Caso 1: A (Cámaras) y C (Vigilancia) deberían estar cerca (Label = 0 para "similar" en algunas implementaciones, o 1 en contrastive standard)
    # Usaremos Label 0 = Similares, Label 1 = Diferentes para standard contrastive logic margin
    
    # A vs C (Similares -> distancia debe ser 0)
    loss_pos = criterion(vec_A, vec_C, label=0) 
    
    # A vs B (Diferentes -> distancia debe ser > margin)
    loss_neg = criterion(vec_A, vec_B, label=1)
    
    total_loss = loss_pos + loss_neg
    
    print(f"Loss Total: {total_loss.item()}")
    print("Sistema listo para indexar licitaciones privadas.")