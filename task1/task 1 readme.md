# Task Service (Java Spring Boot)

Spring Boot REST API to manage and run shell-command-based tasks, persisted in MongoDB.

## Build & Run

Prereqs: Java 17+, Maven, MongoDB running locally or a connection URI.

```bash
# from project root (cloud-cost-calculator)
cd task-service
mvn spring-boot:run
```

Configure Mongo via env var:

```bash
set MONGODB_URI=mongodb://localhost:27017/tasks_db   # Windows PowerShell: $env:MONGODB_URI = "mongodb://localhost:27017/tasks_db"
```

Service runs on http://localhost:8085

## Data Model

Task
- id: String
- name: String
- owner: String
- command: String
- taskExecutions: List<TaskExecution>

TaskExecution
- startTime: Date
- endTime: Date
- output: String

## Endpoints

- GET `/tasks` — returns all tasks
- GET `/tasks?id={id}` — returns single task or 404
- PUT `/tasks` — upsert task (validates command)
- DELETE `/tasks/{id}` — delete task
- GET `/tasks/search?name=foo` — find by name (contains), 404 if none
- PUT `/tasks/{id}/executions` — run the task command and append a TaskExecution

## Command Safety

A simple validator blocks obviously unsafe patterns (e.g., `&&`, `;`, `rm`, `shutdown`, fork bombs, newlines). Adjust `CommandValidator` as needed for your environment.

## Example Requests (curl)

Create/Upsert task:
```bash
curl -X PUT http://localhost:8085/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "id": "123",
    "name": "Print Hello",
    "owner": "John Smith",
    "command": "echo Hello World again!"
  }'
```

Get all tasks:
```bash
curl http://localhost:8085/tasks
```

Get single task:
```bash
curl "http://localhost:8085/tasks?id=123"
```

Search by name:
```bash
curl "http://localhost:8085/tasks/search?name=Print"
```

Run a task execution: 
curl -X PUT http://localhost:8085/tasks/123/executions
```

Delete task:
```bash
curl -X DELETE http://localhost:8085/tasks/123 -i
```

## Notes

- This demo executes commands on the host system. In a real Kubernetes scenario, replace `ShellExecutor` with code that execs into a pod (e.g., Kubernetes Java client) and capture stdout/stderr.
- Make screenshots with Postman/curl showing request and response to satisfy the assignment requirement.
