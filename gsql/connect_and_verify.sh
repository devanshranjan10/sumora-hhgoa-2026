#!/usr/bin/env bash
# ============================================================================
# Sumora - load, verify, and health-check the TigerGraph fraud graph.
#
# Usage (on the CPU VM, or any host with Docker and TigerGraph running):
#   bash gsql/connect_and_verify.sh [load|verify|all]
#
#   load   = apply schema + create loading jobs + run all loads + maintenance
#            queries + algorithm suite (expects prepped CSVs mounted at
#            /home/tigergraph/data_in/HHGOA_IEEE/prepped inside the container)
#   verify = run the PASS/FAIL census checks (fast, read-only)
#   all    = load, then verify (default)
#
# Environment:
#   TG_CONTAINER  container name (default: tigergraph)
# ============================================================================
set -u

CONTAINER="${TG_CONTAINER:-tigergraph}"
MODE="${1:-all}"
GSQL_BIN="/home/tigergraph/tigergraph/app/cmd"

EXPECTED_TXNS=+590742
EXPECTED_CLOSED=5565
EXPECTED_CASES=+20
EXPECTED_PATTERNS=7

PASS=0
FAIL=0

gsql_pipe() {
  # Feed GSQL statements on stdin; print server output.
  # NOTE: only safe for single-line statements (stdin runs line-by-line).
  sudo docker exec -i "$CONTAINER" bash -c "export PATH=\$PATH:${GSQL_BIN}; gsql"
}

run_gsql_file() {
  # run_gsql_file: reads GSQL statements from stdin, writes them to a file,
  # copies into the container and runs `gsql -f` (whole-file mode - safe for
  # multi-line statements).
  local hostf="/tmp/hh_stmt_$$.gsql"
  cat > "$hostf"
  sudo docker cp "$hostf" "$CONTAINER:/tmp/hh_stmt.gsql" >/dev/null
  sudo docker exec "$CONTAINER" bash -c \
    "export PATH=\$PATH:${GSQL_BIN}; gsql -f /tmp/hh_stmt.gsql"
  rm -f "$hostf"
}

ship_and_run() {
  # ship_and_run <local-file.gsql>
  local f="$1"
  sudo docker cp "$f" "$CONTAINER:/tmp/hh_verify_$(basename "$f")"
  sudo docker exec "$CONTAINER" bash -c \
    "export PATH=\$PATH:${GSQL_BIN}; gsql -f /tmp/hh_verify_$(basename "$f")"
}

census_gsql() {
  run_gsql_file <<'EOF'
USE GRAPH SumoraFraudGraph
INTERPRET QUERY () FOR GRAPH SumoraFraudGraph {
  SumAccum<INT> @@next_edges;
  T = {Transaction.*};   PRINT T.size() AS txns;
  N = SELECT t2 FROM T:t1 -(NEXT>:e)- Transaction:t2
      ACCUM @@next_edges += 1;
  PRINT @@next_edges AS next_edges;
  K = {Card.*};          PRINT K.size() AS cards;
  C = {Customer.*};      PRINT C.size() AS customers;
  CC = {ClosedCase.*};   PRINT CC.size() AS closed_cases;
  FC = {FraudCase.*};    PRINT FC.size() AS fraud_cases;
  IC = {IdentityCluster.*}; PRINT IC.size() AS clusters;
  DP = {DeviceProfile.*};   PRINT DP.size() AS devices;
  EM = {EmailAddress.*};    PRINT EM.size() AS emails;
  BR = {BillingRegion.*};   PRINT BR.size() AS regions;
  FP = {FraudPattern.*};    PRINT FP.size() AS patterns;
}
EOF
}

check_count() {
  # check_count <label> <expected|+minimum|-> <json-output> <key>
  local label="$1" expected="$2" json="$3" key="$4"
  local actual
  actual=$(printf '%s' "$json" | grep -oE "\"${key}\": [0-9]+" | grep -oE '[0-9]+$' | head -1)
  if [ -z "$actual" ]; then
    echo "FAIL  ${label}: key '${key}' not found in output"
    FAIL=$((FAIL + 1))
    return
  fi
  if [ "$expected" = "-" ]; then
    if [ "$actual" -gt 0 ]; then
      echo "PASS  ${label}: ${actual}"
      PASS=$((PASS + 1))
    else
      echo "FAIL  ${label}: expected > 0, got ${actual}"
      FAIL=$((FAIL + 1))
    fi
  elif [[ "$expected" == +* ]]; then
    local minimum="${expected#+}"
    if [ "$actual" -ge "$minimum" ]; then
      echo "PASS  ${label}: ${actual} (>= ${minimum})"
      PASS=$((PASS + 1))
    else
      echo "FAIL  ${label}: expected >= ${minimum}, got ${actual}"
      FAIL=$((FAIL + 1))
    fi
  elif [ "$actual" -eq "$expected" ]; then
    echo "PASS  ${label}: ${actual}"
    PASS=$((PASS + 1))
  else
    echo "FAIL  ${label}: expected ${expected}, got ${actual}"
    FAIL=$((FAIL + 1))
  fi
}

do_load() {
  set -e
  HERE="$(cd "$(dirname "$0")" && pwd)"

  echo "== 1/6 schema =="
  ship_and_run "${HERE}/schema.gsql"

  echo "== 2/6 loading jobs =="
  ship_and_run "${HERE}/loading.gsql"

  echo "== 3/6 running loads =="
  cat <<'EOF' | gsql_pipe
USE GRAPH SumoraFraudGraph
RUN LOADING JOB job_load_transactions
RUN LOADING JOB job_load_next
RUN LOADING JOB job_load_txn_edges
RUN LOADING JOB job_load_identity
RUN LOADING JOB job_load_closed_cases
RUN LOADING JOB job_load_closed_case_edges
RUN LOADING JOB job_seed_patterns
RUN LOADING JOB job_load_case_pack
EOF
  echo "(GSQL waits for each loading job to finish before continuing)"

  echo "== 4/6 query definitions =="
  ship_and_run "${HERE}/jobs.gsql"
  ship_and_run "${HERE}/algorithms.gsql"
  ship_and_run "${HERE}/queries.gsql"
  ship_and_run "${HERE}/pattern_scorers.gsql"
  ship_and_run "${HERE}/discovery.gsql"
  echo "== 5/6 install all queries =="
  cat <<'EOF' | gsql_pipe
USE GRAPH SumoraFraudGraph
INSTALL QUERY ALL
EOF

  echo "== 6/6 populate graph features =="
  cat <<'EOF' | gsql_pipe
USE GRAPH SumoraFraudGraph
RUN QUERY job_propagate_case_labels()
RUN QUERY job_build_identity_clusters()
RUN QUERY job_compute_pattern_priors()
RUN QUERY job_precompute_algorithms()
EOF
  bash "${HERE}/install_algorithms.sh"
  cat <<'EOF' | gsql_pipe
USE GRAPH SumoraFraudGraph
RUN QUERY job_run_algorithm_suite()
EOF
  set +e
}

do_verify() {
  echo "== census =="
  local out
  out=$(census_gsql | gsql_pipe)

  check_count "transactions"        "$EXPECTED_TXNS"    "$out" "txns"
  check_count "next edges"         "+575892"         "$out" "next_edges"
  check_count "cards"               "-"                 "$out" "cards"
  check_count "customers"           "-"                 "$out" "customers"
  check_count "closed_cases"        "$EXPECTED_CLOSED"  "$out" "closed_cases"
  check_count "fraud_cases"         "$EXPECTED_CASES"   "$out" "fraud_cases"
  check_count "identity_clusters"   "-"                 "$out" "clusters"
  check_count "device_profiles"     "-"                 "$out" "devices"
  check_count "email_addresses"     "-"                 "$out" "emails"
  check_count "billing_regions"     "-"                 "$out" "regions"
  check_count "fraud_patterns"      "$EXPECTED_PATTERNS" "$out" "patterns"

  echo "== labels + algorithms =="
  local lab
  lab=$(run_gsql_file <<'EOF'
USE GRAPH SumoraFraudGraph
INTERPRET QUERY () FOR GRAPH SumoraFraudGraph {
  T = {Transaction.*};
  F = SELECT t FROM T:t WHERE t.label_confirmed_fraud == true;
  L = SELECT t FROM T:t WHERE t.louvain_community > 0;
  PRINT F.size() AS fraud_labels;
  PRINT L.size() AS louvain_txns;
}
EOF
)
  check_count "fraud-labeled txns"   "-"  "$lab" "fraud_labels"
  check_count "louvain-assigned txns" "-" "$lab" "louvain_txns"

  echo "== installed queries =="
  local qs
  qs=$(printf 'USE GRAPH SumoraFraudGraph\nLS\n' | gsql_pipe | grep -c "q_")
  if [ "$qs" -ge 16 ]; then
    echo "PASS  installed q_* queries: ${qs} (>= 16)"
    PASS=$((PASS + 1))
  else
    echo "FAIL  installed q_* queries: ${qs} (< 16)"
    FAIL=$((FAIL + 1))
  fi

  echo ""
  echo "RESULT: ${PASS} passed, ${FAIL} failed"
  [ "$FAIL" -eq 0 ]
}

case "$MODE" in
  load)   do_load ;;
  verify) do_verify ;;
  all)    do_load; do_verify ;;
  *) echo "usage: $0 [load|verify|all]"; exit 2 ;;
esac
