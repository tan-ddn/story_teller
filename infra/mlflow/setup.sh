#!/bin/sh
# One-time setup of the internal MLflow on the Mac Studio. Safe to re-run.
# Needs .env (copy .env.example and fill in the secrets).
set -eu
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
set -a; . ./.env; set +a
MLFLOW_VERSION=3.17.0
PG=/opt/homebrew/opt/postgresql@17/bin

# Postgres 17, listening on localhost only (the Homebrew default).
brew list postgresql@17 >/dev/null 2>&1 || brew install postgresql@17
brew services start postgresql@17
until "$PG/pg_isready" -q -h 127.0.0.1; do sleep 1; done

# The Mac is shared: replace Homebrew's "trust" with passwords for TCP and peer auth for the socket.
HBA=$("$PG/psql" -d postgres -tAc "show hba_file")
if grep -q "trust" "$HBA"; then
  cat > "$HBA" <<EOF
local   all  all                 peer
host    all  all  127.0.0.1/32   scram-sha-256
host    all  all  ::1/128        scram-sha-256
EOF
  brew services restart postgresql@17
  until "$PG/pg_isready" -q -h 127.0.0.1; do sleep 1; done
fi

# Role and databases (runs and users are kept in separate databases).
"$PG/psql" -d postgres -v ON_ERROR_STOP=1 -q <<EOF
SELECT 'CREATE ROLE mlflow LOGIN' WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'mlflow')\gexec
ALTER ROLE mlflow PASSWORD '${POSTGRES_PASSWORD}';
SELECT 'CREATE DATABASE mlflow OWNER mlflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
SELECT 'CREATE DATABASE mlflow_auth OWNER mlflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_auth')\gexec
EOF

# MLflow, pinned, in its own environment.
uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv "mlflow[auth]==${MLFLOW_VERSION}" "psycopg[binary]==3.3.6"

mkdir -p "$ARTIFACT_DIR"

# launchd agent: start at login, restart on crash.
PLIST="$HOME/Library/LaunchAgents/com.story-teller.mlflow.plist"
sed "s|__DIR__|$PWD|g" com.story-teller.mlflow.plist > "$PLIST"
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "MLflow starting on http://${TAILSCALE_IP}:5000 (log: $PWD/mlflow.log)"
