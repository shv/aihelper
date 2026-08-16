CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id text PRIMARY KEY,
    title text NOT NULL,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536) NOT NULL,
    content_hash text NOT NULL,
    embedding_model text NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
