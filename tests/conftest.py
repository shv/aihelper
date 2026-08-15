import os

if os.getenv("RUN_SMOKE_TESTS") != "1":
    os.environ["OPENAI_API_KEY"] = "unit-test-placeholder"
