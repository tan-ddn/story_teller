# Internal MLflow

Experiment tracking and the model registry for training and evaluating story_teller's models
(Archivist, Scripty, Brainstormer, Showrunner, Censor). It is separate from the dev stack in
`infra/docker` and from anything in production.

It runs natively on the Mac Studio (no Docker) and is reachable by the team over Tailscale at
`http://100.123.117.116:5000`, with a login.

| Part | Where |
|---|---|
| MLflow 3.17.0 server, with basic auth | `uv` environment in this directory, run by a launchd agent |
| Runs, metrics, users | Postgres 17 from Homebrew, localhost only, password required |
| Artifacts (models, adapters, files) | `~/mlflow/artifacts` on the internal disk, served through MLflow |

## Set up

On the Mac Studio, from this directory:

```bash
cp .env.example .env    # then fill in the secrets: openssl rand -hex 24
./setup.sh
```

`setup.sh` installs Postgres, replaces its default passwordless access (the Mac is shared),
creates the databases, installs MLflow and starts it through launchd. It is safe to re-run.

The server starts at login and restarts if it crashes. If the artifacts folder is missing it
refuses to start and retries every minute. Logs are in `mlflow.log`.

```bash
launchctl kickstart -k gui/$(id -u)/com.story-teller.mlflow   # restart
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.story-teller.mlflow.plist   # stop
```

## Log in and add users

The admin account is `admin`, with `MLFLOW_ADMIN_PASSWORD` from `.env`. Signed-in users can read
everything by default. To add a user:

```bash
curl -u admin:$MLFLOW_ADMIN_PASSWORD -X POST http://100.123.117.116:5000/api/2.0/mlflow/users/create \
  -H 'Content-Type: application/json' -d '{"username": "alice", "password": "..."}'
```

## Use from code

```bash
export MLFLOW_TRACKING_URI=http://100.123.117.116:5000
export MLFLOW_TRACKING_USERNAME=alice
export MLFLOW_TRACKING_PASSWORD=...
```

Always log files through MLflow (`mlflow.log_artifact`, `log_model`), never by writing into the
artifacts folder or recording absolute paths. That keeps the move to cloud storage a copy plus a
config change.

**Keep artifacts small.** The internal disk has under 80 GB free. Log metrics, configs, result
files and final LoRA adapters, never base model weights or every checkpoint.

**Why not the 4 TB external drive:** macOS blocks processes started by launchd from removable
volumes ("Operation not permitted"); only SSH sessions have that permission here. To move
artifacts to `/Volumes/Untitled`, grant Full Disk Access at the Mac (System Settings → Privacy &
Security → Full Disk Access → + → ⌘⇧G) to the Python that runs MLflow, found with
`readlink -f .venv/bin/python3`. That path changes when `uv` installs a new Python patch release,
and the permission must then be granted again. Then copy the folder and change `ARTIFACT_DIR`.

## Move to cloud storage later

Artifacts are stored as `mlflow-artifacts:/...` paths relative to `ARTIFACT_DIR`, so moving them
needs no database changes:

1. `aws s3 sync ~/mlflow/artifacts s3://<bucket>/artifacts`, then set
   `ARTIFACT_DIR=s3://<bucket>/artifacts` (the server then needs `boto3` and AWS credentials).
2. `pg_dump -Fc` both `mlflow` and `mlflow_auth`, restore them into the cloud Postgres and point
   the server at it. Keep the same MLflow version for the restore.
3. Clients only change `MLFLOW_TRACKING_URI`.

## Back up

```bash
/opt/homebrew/opt/postgresql@17/bin/pg_dump -Fc mlflow > mlflow.dump   # and mlflow_auth
```

plus the artifacts folder.
