import os

os.environ["MCP_ISSUER_URL"] = "https://auth.example.test/"
os.environ["MCP_RESOURCE_SERVER_URL"] = "http://127.0.0.1:8001/mcp"
os.environ["MCP_REQUIRED_SCOPE"] = "repair:use"

if os.getenv("RUN_SMOKE_TESTS") != "1":
    os.environ["OPENAI_API_KEY"] = "unit-test-placeholder"
