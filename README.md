# AI Helper

Простое приложение на FastAPI.

## Запуск

```bash
pyenv local aihelper
poetry install
poetry run uvicorn main:app --reload
```

После запуска:

- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
