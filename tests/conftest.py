import os

os.environ["MCP_BEARER_TOKEN"] = "unit-test-mcp-bearer-token"
os.environ["MCP_ISSUER_URL"] = "http://127.0.0.1:8001"

if os.getenv("RUN_SMOKE_TESTS") != "1":
    os.environ["OPENAI_API_KEY"] = "unit-test-placeholder"
