CREATE EXTENSION IF NOT EXISTS pg_textsearch;

CREATE INDEX IF NOT EXISTS document_chunks_bm25_idx
ON document_chunks
USING bm25 ((title || E'\n' || content))
WITH (
    text_config = 'russian',
    k1 = 1.2,
    b = 0.75
);
