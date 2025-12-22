CREATE EXTENSION IF NOT EXISTS vector;      
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- Para IDs únicos de auditoría
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- Para búsqueda borrosa de texto

-- ===========================
--REGISTRO DE LA LICITACIÓN 
-- ===========================
CREATE TABLE IF NOT EXISTS registro_licitaciones (
    id                  BIGSERIAL PRIMARY KEY,
    codigo_proceso      VARCHAR(255) UNIQUE NOT NULL,
    entidad  VARCHAR(255), 
    -- Metadata de tiempo y volumen
    fecha_hora_ingesta  TIMESTAMPTZ DEFAULT NOW(),
    numero_pliegos      INT DEFAULT 0,                
    
    -- Estado del procesamiento global
    estado_actual       VARCHAR(50) DEFAULT 'INGESTA', -- 'INGESTA', 'PROCESANDO', 'INDEXADO', 'ERROR'
    
    -- Metadata Global (Taxonomía inferida, cuantía total, etc.)
    metadata_global     JSONB DEFAULT '{}'::jsonb
);

-- ======================================
-- REGISTRO DE ARCIHVOS (Los PDFs Crudos)
-- ======================================
CREATE TABLE IF NOT EXISTS registro_pdfs (
    id                  BIGSERIAL PRIMARY KEY,
    licitacion_id       BIGINT REFERENCES registro_licitaciones(id) ON DELETE CASCADE,
    
    nombre_archivo      VARCHAR(255) NOT NULL,
    ruta_almacenamiento TEXT, -- Path en S3 o Local
    
    -- Metadata técnica del archivo
    metadata_archivo    JSONB, -- { "pages": 40, "size_mb": 2.5, "sha256": "..." }
    
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ======================================
-- SECCIONES ESTRUCTURADAS (El Contexto Semántico)
-- Esta tabla rompe el PDF en "Capítulos" (Financiero, Técnico, etc.)
-- ======================================
CREATE TABLE IF NOT EXISTS secciones_documento (
    id                  BIGSERIAL PRIMARY KEY,
    pdf_id              BIGINT REFERENCES registro_pdfs(id) ON DELETE CASCADE,
    
    titulo_detectado    TEXT,        -- Ej: "3.2 CAPACIDAD FINANCIERA"
    categoria_seccion   VARCHAR(50), -- 'JURIDICO', 'FINANCIERO', 'EXPERIENCIA', 'TECNICO'
    
    -- Metadata "Final" extraída por el LLM para esta sección completa
    -- Aquí guardas el JSON grande de requisitos de esta sección
    metadata_extracted  JSONB,       -- { "requisitos": [ ... ] }
    
    page_start          INT,         -- Para referencia visual
    page_end            INT
);

-- Índice para buscar rápido dentro del JSONB (Ej: buscar secciones con 'liquidez')
CREATE INDEX idx_secciones_meta ON secciones_documento USING GIN (metadata_extracted);

-- =========================================================================
-- NIVEL 4: NODOS VECTORIZADOS (Los Átomos del Grafo)
-- Aquí viven tus Chunks de texto y tus Términos especiales.
-- =========================================================================
CREATE TABLE IF NOT EXISTS nodos_vectorizados (
    id                  BIGSERIAL PRIMARY KEY,
    seccion_id          BIGINT REFERENCES secciones_documento(id) ON DELETE CASCADE,
    
  
    tipo_nodo           VARCHAR(50), -- 'CHUNK_TEXTO', 'CONCEPTO_CLAVE', 'REQUISITO_ATOMICO'
    
    -- Contenido
    contenido_texto     TEXT,        --
    
    -- Metadata específica del nodo
    -- Ej: { "operador": ">=", "valor": 1.5, "unidad": "veces" }
    metadata_nodo       JSONB, 
    embedding_vec       vector(768) 
);

-- Índices para búsqueda vectorial rápida (Similitud Coseno)
CREATE INDEX idx_nodos_vec ON nodos_vectorizados USING ivfflat (embedding_vec vector_cosine_ops) WITH (lists = 100);

-- =========================================================================
-- NIVEL 5: AUDITORÍA Y LOGS (El Cerebro de Entrenamiento)
-- Aquí registras el cálculo del Score y la Contrastive Loss
-- =========================================================================
CREATE TABLE IF NOT EXISTS logs_auditoria (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    licitacion_id       BIGINT REFERENCES registro_licitaciones(id),
    
    evento              VARCHAR(50), -- 'CALCULO_SCORE', 'TRAINING_STEP', 'ERROR_API'
    fecha_evento        TIMESTAMPTZ DEFAULT NOW(),
    -- Detalles técnicos
    detalles            JSONB, -- { "contrastive_loss": 0.045, "score_final": 89.5, "model_ver": "v2" }   
    -- Opcional: Si quieres auditar qué usuario/empresa gatilló el evento
    usuario_trigger     VARCHAR(100)
);


CREATE INDEX idx_logs_evento ON logs_auditoria(evento);
CREATE INDEX idx_logs_fecha ON logs_auditoria(fecha_evento);