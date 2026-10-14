#!/bin/sh
set -e
mkdir -p "${DATA_DIR:-/data}"
if [ "$(id -u)" = "0" ]; then
    # Fly mounts a fresh volume owned by root: hand it to the app user, then drop root
    chown -R appuser:appuser "${DATA_DIR:-/data}"
    exec gosu appuser "$@"
fi
# already started as the app user (e.g. docker run --user 1000)
exec "$@"
