"""Run: `poetry run python -m scripts.run_mcp_http`"""

from app.mcp.server import mcp


def main() -> None:
    mcp.run(
        "streamable-http",
        host="127.0.0.1",
        port=8001,
        streamable_http_path="/mcp",
        json_response=False,
        stateless_http=False,
        max_request_body_size=4 * 1024 * 1024,
        session_idle_timeout=1800.0,
        max_sessions=100,
    )


if __name__ == "__main__":
    main()
