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


# Команды для Postgres

## Применение миграции

```
docker compose exec -T postgres \
  psql -U aihelper -d aihelper \
  < db/002_bm25.sql
```

## Запрос в базу

```
docker compose exec postgres \
  psql -U aihelper -d aihelper \
  -c "
    SELECT name, default_version, installed_version
    FROM pg_available_extensions
    WHERE name IN ('vector', 'pg_textsearch')
    ORDER BY name;
  "
```