# YOLO Service SQLAlchemy Refactor Documentation

## 1. Introduction
This document explains the SQLAlchemy refactor done in the YOLO service, why it was useful, what changed in the code, and how skills and evals work in this project.

## 2. Why SQLAlchemy is better than using SQLite directly
SQLite is a simple embedded database that is good for small projects and local development. SQLAlchemy provides a more structured and scalable way to work with databases.

### Benefits of SQLAlchemy
- Object-relational mapping (ORM): database tables are represented as Python classes.
- Cleaner code: developers work with Python objects instead of writing many raw SQL queries.
- Better maintainability: database logic is separated from API logic.
- Database flexibility: the same code can work with SQLite for development and PostgreSQL for production.
- Safer and more reusable queries: ORM queries reduce manual SQL string errors.
- Easier future extensions: adding tables or new columns becomes simpler.

### SQLite limitations
- Raw SQL is scattered through the application code.
- It is less maintainable for larger systems.
- It is harder to switch databases later.
- It requires more manual handling of connections and queries.

## 3. What changed in the YOLO project
The YOLO service was refactored so that the database layer is handled in a cleaner and more professional way.

### Before
- The application used raw SQLite logic directly inside the API service.
- Database operations were mixed with request handling.
- The code was less structured.

### After
- The database model was moved into SQLAlchemy model classes.
- Database sessions were created in a dedicated database module.
- FastAPI endpoints now use dependency injection to access the DB session.
- The public API behavior stayed the same.

## 4. Main files changed
### services/yolo/models.py
This file defines the database tables as Python classes.

- PredictionSession: stores information about each prediction request.
- DetectionObject: stores each detected object from the prediction result.

### services/yolo/db.py
This file handles:
- database engine creation
- session creation
- database initialization
- database backend selection between SQLite and PostgreSQL

### services/yolo/app.py
The API endpoints were updated to use SQLAlchemy instead of manual SQL statements.

- data is inserted using ORM objects
- data is queried using ORM filters
- the API behavior remains the same

## 5. What the refactor achieved
The refactor improved the project by:
- separating database logic from business logic
- making the code easier to read
- making future maintenance easier
- allowing database switching with environment variables
- preparing the project for production-style deployment

## 6. What is a skill?
A skill is a reusable instruction file that tells an AI coding agent how to perform a specific task correctly.

In this project, the skill is stored in:
- .agents/skills/data-layer/SKILL.md

### Purpose of the skill
The skill explains:
- when to use the data layer approach
- what files to modify
- what patterns to follow
- what to avoid
- how to preserve existing behavior

### Example use cases
The skill is used for tasks such as:
- refactoring to SQLAlchemy
- adding a new endpoint that reads from the database
- adding a new table
- changing the database backend to PostgreSQL

## 7. What are evals?
Evals are evaluation cases used to test whether a skill works properly.

They are stored in:
- .agents/skills/data-layer/evals/evals.json

### Why evals are important
Evals help ensure that:
- the skill gives the correct guidance
- the agent changes the right files
- the implementation follows the expected pattern
- unwanted patterns are avoided

### What an eval contains
Each eval usually includes:
- a prompt that a developer might type
- an expected output description
- checks for what the agent should or should not do

## 8. How skills and evals work together
- The skill provides instructions for the agent.
- The evals check whether the agent followed those instructions correctly.
- Together, they make the agent more reliable and consistent.

## 9. Summary
This assignment showed how to modernize a backend service by moving from direct SQLite handling to SQLAlchemy ORM. The result is a cleaner, more maintainable, and more scalable architecture. Skills and evals help future agents perform similar tasks more reliably and consistently.
