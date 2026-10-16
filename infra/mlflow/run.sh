#!/bin/sh
# Start the MLflow tracking server. launchd runs this (see com.story-teller.mlflow.plist).
# Writes the basic-auth config from .env (MLflow's ini file cannot read env vars), then execs mlflow.
set -eu
cd "$(dirname "$0")"
set -a; . ./.env; set +a

# Refuse to start if the artifacts folder is missing (e.g. an unmounted drive).
if [ ! -d "$ARTIFACT_DIR" ]; then
  echo "artifact dir $ARTIFACT_DIR missing (not mounted?)" >&2
  exit 1
fi

DB=postgresql://mlflow:${POSTGRES_PASSWORD}@127.0.0.1:5432
umask 077
cat > basic_auth.ini <<EOF
[mlflow]
default_permission = READ
database_uri = ${DB}/mlflow_auth
admin_username = admin
admin_password = ${MLFLOW_ADMIN_PASSWORD}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
EOF
export MLFLOW_AUTH_CONFIG_PATH="$PWD/basic_auth.ini"
export MLFLOW_FLASK_SERVER_SECRET_KEY="$MLFLOW_SECRET_KEY"

exec ./.venv/bin/mlflow server \
  --host "$TAILSCALE_IP" --port 5000 \
  --app-name basic-auth \
  --backend-store-uri "${DB}/mlflow" \
  --artifacts-destination "$ARTIFACT_DIR" \
  --allowed-hosts "${TAILSCALE_IP}:5000"
