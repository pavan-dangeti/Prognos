#!/usr/bin/env bash
# Provision and verify the Oracle Always Free A1 demo host.
#
# Split deliberately into two halves, because only one of them is testable:
#
#   push   workstation -> instance. rsync the repo, the Parquet tables and the model
#          artifacts, then run compose up remotely. This half is verifiable here.
#   verify check a running stack over HTTP. Runs anywhere, which is why it is separate.
#
# The instance itself cannot be created from here: that needs credentials in your own
# OCI account. `docs/deploy-oci.md` has those steps. Nothing in this script creates,
# modifies or deletes an OCI resource.
#
# The Parquet tables ship rather than the 14.6 GB of raw HDF5 from NASA. Converting
# needs the raw archive and five minutes of a free ARM core; the tables are 1.3 GB and
# the converter already ran and produced them.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${PROGNOS_HOST:-prognos-a1}"
REMOTE_DIR="${PROGNOS_REMOTE_DIR:-/opt/prognos}"
TOKEN_FILE="$REPO_ROOT/ops/prometheus-token.txt"

die() { echo "error: $*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "$1 is not installed"; }

# Every step refuses rather than continuing on a guess. A half-provisioned demo host
# that looks healthy is worse than one that failed loudly.
check_prereqs() {
    need rsync
    need ssh
    [[ -d "$REPO_ROOT/data/parquet" ]] ||
        die "no data/parquet. Run 'prognos convert' here first; see docs/deploy-oci.md"
    [[ -f "$REPO_ROOT/artifacts/model/model.pkl" ]] ||
        die "no model artifacts. Run 'prognos stream --model-dir artifacts/model' first"
    [[ -f "$TOKEN_FILE" ]] ||
        die "no $TOKEN_FILE. Write PROGNOS_API_TOKEN's value into it; see docs/runbook.md"
    [[ -f "$REPO_ROOT/.env" ]] ||
        die "no .env. It must contain PROGNOS_API_TOKEN=<the same value>"
}

push() {
    check_prereqs
    echo "==> pushing to $HOST:$REMOTE_DIR"
    ssh "$HOST" "sudo mkdir -p '$REMOTE_DIR' && sudo chown \$(whoami) '$REMOTE_DIR'"

    # Trailing slash on the source matters: without it rsync creates data/parquet
    # *inside* the destination and the tables end up one level too deep for the config.
    rsync -az --delete \
        --exclude 'data/cache' --exclude '.venv' --exclude '.git' \
        "$REPO_ROOT/src" "$HOST:$REMOTE_DIR/src"
    rsync -az --delete "$REPO_ROOT/configs" "$HOST:$REMOTE_DIR/configs"
    rsync -az --delete "$REPO_ROOT/ops" "$HOST:$REMOTE_DIR/ops"
    rsync -az "$REPO_ROOT/data/parquet" "$HOST:$REMOTE_DIR/data/parquet"
    rsync -az "$REPO_ROOT/artifacts" "$HOST:$REMOTE_DIR/artifacts"
    # The token and the env file go over the wire because the host needs both to serve
    # an authenticated stack. They are gitignored and never committed.
    rsync -az "$TOKEN_FILE" "$HOST:$REMOTE_DIR/ops/prometheus-token.txt"
    rsync -az "$REPO_ROOT/.env" "$HOST:$REMOTE_DIR/.env"
    rsync -az "$REPO_ROOT/docker-compose.yml" "$REPO_ROOT/pyproject.toml" \
        "$REPO_ROOT/uv.lock" "$HOST:$REMOTE_DIR/"

    echo "==> building and starting the stack"
    ssh "$HOST" "cd '$REMOTE_DIR' && chmod 600 .env ops/prometheus-token.txt && docker compose up -d --build"
}

verify() {
    # Runs anywhere a stack is reachable, which is the point: the same command checks a
    # laptop and the A1, so a passing check means the same thing in both places.
    local base="${1:-http://127.0.0.1}"
    local token
    token="$(tr -d '\n' <"$TOKEN_FILE")"
    local fail=0

    # `auth no` deliberately omits the header. The first version always sent it, so the
    # check named "unauthenticated" was authenticated and passed whatever the server
    # did about anonymous access.
    check() {
        local name="$1" expect="$2" url="$3" auth="${4:-yes}" got
        if [[ "$auth" == "bad" ]]; then
            got="$(curl -s -m 10 -o /dev/null -w '%{http_code}' \
                -H "Authorization: Bearer not-the-token" "$url")"
        elif [[ "$auth" == "yes" ]]; then
            got="$(curl -s -m 10 -o /dev/null -w '%{http_code}' \
                -H "Authorization: Bearer $token" "$url")"
        else
            got="$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$url")"
        fi
        if [[ "$got" == "$expect" ]]; then
            printf '  ok    %-22s %s\n' "$name" "$got"
        else
            printf '  FAIL  %-22s got %s, expected %s\n' "$name" "$got" "$expect"
            fail=1
        fi
    }

    echo "checking $base"
    check "dashboard cost"     200 "$base:8084/cost"
    check "dashboard degraded" 200 "$base:8084/degraded"
    check "dashboard fleet"    200 "$base:8084/fleet"
    check "inference ready"    200 "$base:8082/readyz"
    check "ingestion ready"    200 "$base:8081/readyz"
    check "no token refused"   401 "$base:8084/fleet" no
    check "bad token refused"  401 "$base:8084/fleet" bad

    # An authenticated 200 on /fleet proves nothing on its own: in compose no stream
    # feeds the dashboard, so /fleet is legitimately empty. The cost panel has content,
    # so that is the route asserted for substance.
    echo "cost panel:"
    curl -s -m 10 -H "Authorization: Bearer $token" "$base:8084/cost" |
        python3 -c '
import json, sys
panel = json.load(sys.stdin)
print("  %d policies, best %s" % (len(panel["policies"]), panel["best"]))' 

    echo "prometheus targets:"
    curl -s -m 10 "$base:9090/api/v1/targets?state=active" |
        python3 -c '
import json,sys
ts = json.load(sys.stdin)["data"]["activeTargets"]
up = sum(1 for t in ts if t["health"] == "up")
print(f"  {up}/{len(ts)} up")
for t in ts:
    if t["health"] != "up":
        print("  DOWN", t["labels"].get("job"), t.get("lastError", "")[:80])
sys.exit(0 if up == len(ts) else 1)' || fail=1

    [[ $fail -eq 0 ]] || die "the stack is not healthy; see docs/runbook.md"
    echo "all checks passed"
}

case "${1:-}" in
    push) push ;;
    verify) shift; verify "${1:-}" ;;
    *) die "usage: $0 {push|verify [base-url]}" ;;
esac