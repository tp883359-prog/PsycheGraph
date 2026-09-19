#!/usr/bin/env bash
# Backup the PsycheGraph demo state that cannot be recreated.
#
# What is included:
#   * PostgreSQL  - conversations, checkpoints, run history (pg_dump -Fp | gzip)
#   * Chroma      - the local vector store directory, tar.gz
#
# What is deliberately NOT included:
#   * .env and any other secret - the backups never read that file
#   * the Hugging Face model cache - it is 2.2 GB and re-downloadable
#   * docs/evaluation_summary.json - it is versioned in git
#
# Usage (from the repository root, with the stack running):
#   ./scripts/backup_demo.sh
#   OUT_DIR=/srv/backups COMPOSE_FILE=docker/docker-compose.yml ./scripts/backup_demo.sh
#
# Restore:
#   gunzip -c backups/postgres-<stamp>.sql.gz | \
#     docker compose -f docker/docker-compose.yml exec -T langgraph-postgres \
#       psql -U postgres -d postgres
#   tar -xzf backups/vectorstore-<stamp>.tar.gz -C data
#   docker compose -f docker/docker-compose.yml restart langgraph-api
#
# Retention: keep the last few archives somewhere off the VPS (see
# docs/PRODUCTION_DEPLOYMENT.md, "Backup"). This script never deletes anything.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE_FILE="${COMPOSE_FILE:-docker/docker-compose.yml}"
OUT_DIR="${OUT_DIR:-backups}"
VECTORSTORE_DIR="${VECTORSTORE_DIR:-data/vectorstore}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

log() { printf '%s\n' "$*"; }

command -v docker >/dev/null 2>&1 || {
	log "docker is required" >&2
	exit 1
}
[[ -f "$COMPOSE_FILE" ]] || {
	log "compose file not found: $COMPOSE_FILE" >&2
	exit 1
}

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

pg_target="$OUT_DIR/postgres-$STAMP.sql.gz"
vs_target="$OUT_DIR/vectorstore-$STAMP.tar.gz"

log "backup: PostgreSQL -> $pg_target"
docker compose -f "$COMPOSE_FILE" exec -T langgraph-postgres \
	pg_dump -U postgres -d postgres | gzip >"$pg_target"

if [[ -d "$VECTORSTORE_DIR" ]]; then
	log "backup: vector store -> $vs_target"
	tar -czf "$vs_target" -C "$(dirname "$VECTORSTORE_DIR")" "$(basename "$VECTORSTORE_DIR")"
else
	log "warning: $VECTORSTORE_DIR does not exist; skipping the index"
	vs_target=""
fi

# Safety net: a secret must never end up in an archive. The paths above cannot
# contain .env, but a mis-set VECTORSTORE_DIR could point somewhere unexpected.
for archive in "$pg_target" ${vs_target:+"$vs_target"}; do
	if [[ "$archive" == *.tar.gz ]] && tar -tzf "$archive" 2>/dev/null | grep -qE '(^|/)\.env$'; then
		log "ERROR: $archive appears to contain .env - deleting it" >&2
		rm -f "$archive"
		exit 1
	fi
	if [[ "$(stat -c %s "$archive")" -eq 0 ]]; then
		log "ERROR: $archive is empty" >&2
		exit 1
	fi
	log "ok: $(basename "$archive") $(du -h "$archive" | cut -f1) sha256=$(sha256sum "$archive" | cut -c1-12)"
done

log ""
log "Backups written to $OUT_DIR."
log "Copy them off the VPS (scp/rsync) - they are the only state that is not reproducible."
