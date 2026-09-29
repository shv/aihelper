"""Run: `poetry run python -m scripts.run_mcp_http`"""

import os

from app.mcp.server import mcp


def main() -> None:
    mcp_host_ip = os.environ.get("MCP_HOST_IP", "127.0.0.1")
    mcp_port = os.environ.get("MCP_PORT", "8001")
    mcp.run(
        "streamable-http",
        host=mcp_host_ip,
        port=int(mcp_port),
        streamable_http_path="/mcp",
        json_response=False,
        stateless_http=False,
        max_request_body_size=4 * 1024 * 1024,
        session_idle_timeout=1800.0,
        max_sessions=100,
    )


if __name__ == "__main__":
    main()
