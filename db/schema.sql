-- ─────────────────────────────────────────────────────────────────────────────
-- BIDV Report AI Agent — PostgreSQL Schema
-- ─────────────────────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Templates ────────────────────────────────────────────────────────────────
-- Stores the .docx template files + metadata
CREATE TABLE templates (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        VARCHAR(255) NOT NULL,           -- "Mẫu BC gửi CQUBKTĐU"
    description TEXT,
    filename    VARCHAR(500) NOT NULL,           -- original filename
    file_data   BYTEA NOT NULL,                  -- raw .docx binary
    file_size   INTEGER,
    version     INTEGER DEFAULT 1,
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Template Fields ───────────────────────────────────────────────────────────
-- Yellow-highlighted fields extracted from each template
CREATE TABLE template_fields (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    template_id  UUID NOT NULL REFERENCES templates(id) ON DELETE CASCADE,
    field_key    VARCHAR(100) NOT NULL,           -- "para_5"
    para_idx     INTEGER NOT NULL,
    placeholder  TEXT NOT NULL,                   -- original yellow text
    context      TEXT,                            -- surrounding paragraph text
    field_order  INTEGER DEFAULT 0,
    embedding    vector(1536),                    -- for semantic search (optional)
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(template_id, field_key)
);

CREATE INDEX idx_template_fields_template ON template_fields(template_id);
CREATE INDEX idx_template_fields_embedding ON template_fields
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

-- ── Source Documents ─────────────────────────────────────────────────────────
-- Uploaded Word/PDF source files per pipeline run
CREATE TABLE source_documents (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename     VARCHAR(500) NOT NULL,
    file_type    VARCHAR(20),                     -- pdf | docx | txt
    file_size    INTEGER,
    extracted_text TEXT,                          -- pandoc/pdftotext output
    embedding    vector(1536),                    -- full-doc embedding
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- ── Pipeline Runs ────────────────────────────────────────────────────────────
-- One row per user-initiated pipeline execution
CREATE TABLE pipeline_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    template_id     UUID NOT NULL REFERENCES templates(id),
    status          VARCHAR(50) DEFAULT 'created',
    -- status values: created | analyzing | extracting | writing | reviewing | approved | exporting | done | error
    current_step    INTEGER DEFAULT 0,            -- 1-5
    langgraph_state JSONB,                        -- full LangGraph state snapshot
    error_message   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_runs_template ON pipeline_runs(template_id);
CREATE INDEX idx_runs_status   ON pipeline_runs(status);

-- ── Run Source Documents (many-to-many) ──────────────────────────────────────
CREATE TABLE run_source_documents (
    run_id  UUID REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    doc_id  UUID REFERENCES source_documents(id) ON DELETE CASCADE,
    PRIMARY KEY (run_id, doc_id)
);

-- ── Field Results ─────────────────────────────────────────────────────────────
-- AI-generated (and human-approved) value for each field in a run
CREATE TABLE field_results (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id          UUID NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    field_id        UUID NOT NULL REFERENCES template_fields(id),
    field_key       VARCHAR(100) NOT NULL,
    ai_value        TEXT,                         -- Claude's raw output
    final_value     TEXT,                         -- after human edit
    confidence      VARCHAR(10),                  -- high | mid | low
    reason          TEXT,                         -- Claude's explanation
    source_refs     JSONB,                        -- which source docs were used
    human_edited    BOOLEAN DEFAULT FALSE,
    approved        BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(run_id, field_key)
);

CREATE INDEX idx_field_results_run ON field_results(run_id);

-- ── Exported Documents ────────────────────────────────────────────────────────
CREATE TABLE exported_documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id      UUID NOT NULL REFERENCES pipeline_runs(id),
    filename    VARCHAR(500),
    file_data   BYTEA,
    export_mode VARCHAR(20),                      -- clean | tracked
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Auto-update updated_at ────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_templates_updated
    BEFORE UPDATE ON templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_runs_updated
    BEFORE UPDATE ON pipeline_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_results_updated
    BEFORE UPDATE ON field_results
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
