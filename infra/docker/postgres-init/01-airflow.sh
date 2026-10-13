#!/bin/sh
# Airflow's metadata lives in its own database, owned by its own role.
set -eu
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -v pw="$AIRFLOW_DB_PASSWORD" <<'SQL'
CREATE ROLE airflow LOGIN PASSWORD :'pw';
CREATE DATABASE airflow OWNER airflow;
SQL
