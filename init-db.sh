#!/bin/sh
set -e

# Langgraph setup
psql -v ON_ERROR_STOP=1 --username "postgres" <<-EOSQL
CREATE USER langgraph_user WITH PASSWORD 'langgraph_password';
CREATE DATABASE langgraph_db OWNER langgraph_user;
GRANT ALL PRIVILEGES ON DATABASE langgraph_db TO langgraph_user;
\c langgraph_db
GRANT ALL ON SCHEMA public TO langgraph_user;
GRANT CREATE ON SCHEMA public TO langgraph_user;
EOSQL

# MLflow setup
psql -v ON_ERROR_STOP=1 --username "postgres" <<-EOSQL
CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';
CREATE DATABASE mlflow_db OWNER mlflow_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
\c mlflow_db
GRANT ALL ON SCHEMA public TO mlflow_user;
GRANT CREATE ON SCHEMA public TO mlflow_user;
EOSQL
