#!/usr/bin/env bash
set -euo pipefail

container="${TG_CONTAINER:-tigergraph}"
cd "$(dirname "$0")/.."

sudo docker exec -i "$container" bash -s <<'INNER'
set -euo pipefail
root="/home/tigergraph/tigergraph/app/4.2.5/lib/gsql-graph-algorithms/algorithms"
gsql="/home/tigergraph/tigergraph/app/cmd/gsql"
while read -r name source; do
  {
    printf 'USE GRAPH SumoraFraudGraph\n'
    cat "$root/$source"
    printf '\nINSTALL QUERY %s\n' "$name"
  } > "/tmp/sumora-${name}.gsql"
  "$gsql" -f "/tmp/sumora-${name}.gsql"
done <<'QUERIES'
tg_louvain Community/louvain/tg_louvain.gsql
tg_lcc Community/local_clustering_coefficient/tg_lcc.gsql
tg_pagerank Centrality/pagerank/global/unweighted/tg_pagerank.gsql
QUERIES
INNER

sudo docker cp scripts/algos_run.gsql "$container:/tmp/sumora-algos-run.gsql"
sudo docker exec "$container" /home/tigergraph/tigergraph/app/cmd/gsql -f /tmp/sumora-algos-run.gsql
