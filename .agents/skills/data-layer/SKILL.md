---
name: yolo-api-data-layer
description: >-
  The authoritative playbook for the YOLO service's API data/persistence layer
  (services/yolo). Use this whenever a task touches how the FastAPI service
  stores or reads data: migrating from raw sqlite3 to SQLAlchemy, adding or
  changing endpoints that query the database, adding or altering tables/columns,
  writing or fixing tests for those endpoints, deleting records, or making the
  database backend configurable (SQLite in dev, Postgres in prod). Trigger this
  even when the request doesn't say "SQLAlchemy" or "data layer" explicitly —
  e.g. "refactor the api to use sqlalchemy", "add an endpoint GET
  /predictions/recent", "add a UserFeedback table", "add a column
  processing_time_ms", "delete a prediction session and its detection objects by
  uid", "the database layer doesn't follow our architectural design, fix it",
  "make the backend configurable so we can use postgres", or "write tests for
  the /predict endpoint". If the change reads from or writes to the YOLO
  service's database, this skill applies.
---

# YOLO API Data Layer

This skill governs the persistence layer of the YOLO object-detection service at
`services/yolo`. The service is FastAPI + Ultralytics; it stores a
`PredictionSession` per uploaded image and one `DetectionObject` per detected
box. Historically this was raw `sqlite3` strings scattered through `app.py`. The
target architecture is **SQLAlchemy ORM** so the same code runs on SQLite in
development and Postgres in production by flipping one environment variable.

Read this whole file first. For the exact per-endpoint translations, the test
migration, and recipes for new features, open the reference files called out
below — don't reconstruct them from memory, they encode behaviour that the
existing tests pin down.

## The one rule that outranks everything

**Existing public behaviour must not change.** Every endpoint keeps the same
path, same HTTP status codes, and the same JSON response shape (same keys, same
types, same nesting). The refactor is internal plumbing only. The test suite in
`services/yolo/tests/` is the contract — if a change makes a test fail, the
change is wrong, not the test (the one exception is migrating the tests
themselves off the old `sqlite3` API, covered in `references/testing.md`).

## Target architecture

Three concerns, three places:

- **`services/yolo/models.py`** — declarative SQLAlchemy models. Tables are
  created from these definitions, so no hand-written `CREATE TABLE`.
- **`services/yolo/db.py`** — the engine, SessionLocal factory, `get_db()`
  generator dependency, and `init_db()` that calls `Base.metadata.create_all`.
  Backend is chosen here from `DB_BACKEND` (sqlite default, postgres opt-in).
- **`services/yolo/app.py`** — endpoints only. Each endpoint that touches the DB
  declares `db: Session = Depends(get_db)` and uses the ORM. No `import
  sqlite3`, no SQL strings, no connection management in `app.py`.

Column types must mirror the current SQLite schema exactly:

| Table | Column | Type | Notes |
|---|---|---|---|
| `prediction_sessions` | `uid` | `String` primary key | |
| | `timestamp` | `DateTime`, `default=datetime.utcnow` | |
| | `original_image` | `String` | |
| | `predicted_image` | `String` | |
| `detection_objects` | `id` | `Integer` primary key, autoincrement | |
| | `prediction_uid` | `String` | links to a session uid |
| | `label` | `String` | |
| | `score` | `Float` | |
| | `box` | `String` | stringified list e.g. `"[10, 20, 100, 200]"` |

Recreate the three indexes from the old `init_db()` (`idx_prediction_uid`,
`idx_label`, `idx_score`) via `Index(...)` in `__table_args__` on
`DetectionObject`.

## The refactor, step by step

1. **Write `models.py`** with `PredictionSession` and `DetectionObject` per the
   table above, plus the indexes. `Base = declarative_base()`.
2. **Write `db.py`**: read env vars, build `DATABASE_URL`, create engine with
   `connect_args={"check_same_thread": False}` only when SQLite; make
   `SessionLocal`; add `init_db()` and `get_db()`.
3. **Rewrite `app.py` endpoints** using `references/endpoint-migration.md`.
4. **Remove dead code**: old `init_db`, `save_prediction_session`,
   `save_detection_object`, `import sqlite3`, `DB_PATH`.
5. **Migrate the tests** following `references/testing.md`.
6. **Update `requirements.txt`**: add `sqlalchemy>=2.0` and `psycopg2-binary`.

For adding columns, new tables, new endpoints, deletes, or Postgres config see
`references/extensions.md`.

## Gotchas that will bite you

- **`timestamp` serialization.** SQLAlchemy `DateTime` returns a Python
  `datetime`; FastAPI serializes it as ISO 8601 with a `T` separator instead of
  a space. Tests don't assert on it so they stay green, but flag it to the user.
- **`box` stays a string.** Keep `box = Column(String)`; do not convert to JSON
  or a list.
- **Empty-label guard.** `if not label.strip():` must stay before any query in
  `/predictions/label/{label}`.
- **`min_score` bounds.** Keep the 0.0–1.0 guard before querying.
- **`prediction_uid` in score response.** Only `/predictions/score/{min}` returns
  `prediction_uid` per object — don't homogenize.
- **`check_same_thread` is SQLite-only.** Gate it on the URL.
- **Commit writes.** `db.add(...)` then `db.commit()`. The ORM does not
  auto-commit.

## Verify before claiming done

- `cd services/yolo && pytest tests/` → all tests pass.
- `python app.py` starts without errors.
- `grep -rnE "sqlite3|conn\.execute|CREATE TABLE|INSERT INTO|SELECT " app.py`
  returns nothing.
- Coverage hasn't regressed.

## Helpful companion skills

- `obra/superpowers@writing-skills` — when editing this skill itself.
- `obra/superpowers@verification-before-completion` — enforces running tests
  before declaring done.
- `anthropics/skills@webapp-testing` — patterns for FastAPI tests.
