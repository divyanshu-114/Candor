#!/usr/bin/env bash
# Run a command with .env exported (.env wins over inherited variables).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a; . ./.env; set +a
exec "$@"
