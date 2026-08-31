#!/usr/bin/env bash
# Run the same checks CI runs, locally, before you push.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Interface conformance tests =="
pytest tests/test_interface_conformance.py -v

echo
echo "== Metrics unit tests =="
pytest tests/test_metrics.py -v

echo
echo "== Full harness run on the toy set =="
python -m harness.run_harness \
  --corpus data/toy/corpus.jsonl \
  --queries data/toy/queries_dev.tsv \
  --qrels data/toy/qrels_dev.txt \
  --baseline-run data/toy/reference_bm25_run_dev.trec \
  --run-out runs/dev_run.trec \
  --report-out runs/dev_report.json

echo
echo "== Full harness run on the full set =="
python -m harness.run_harness \
  --corpus data/full/corpus.jsonl \
  --queries data/full/queries_dev.tsv \
  --qrels data/full/qrels_dev.txt \
  --baseline-run data/full/reference_bm25_run_dev.trec \
  --run-out runs/full_run.trec \
  --report-out runs/full_report.json

echo
echo "== Full harness run on cranfield =="
python -m harness.run_harness \
  --corpus data/cranfield/corpus.jsonl \
  --queries data/cranfield/queries_dev.tsv \
  --qrels data/cranfield/qrels_dev.txt \
  --run-out runs/cranfield_run.trec \
  --report-out runs/cranfield_report.json

echo
echo "== Full harness run on cacm =="
python -m harness.run_harness \
  --corpus data/cacm/corpus.jsonl \
  --queries data/cacm/queries_dev.tsv \
  --qrels data/cacm/qrels_dev.txt \
  --run-out runs/cacm_run.trec \
  --report-out runs/cacm_report.json

run_proxy() {
  local name="$1" dir="$2"
  echo
  echo "== Full harness run on ${name} =="
  if [[ ! -f "${dir}/corpus.jsonl" || ! -f "${dir}/queries_dev.tsv" || ! -f "${dir}/qrels_dev.txt" ]]; then
    echo "SKIP (missing ${dir})"
    return 0
  fi
  python -m harness.run_harness \
    --corpus "${dir}/corpus.jsonl" \
    --queries "${dir}/queries_dev.tsv" \
    --qrels "${dir}/qrels_dev.txt" \
    --run-out "runs/${name}_run.trec" \
    --report-out "runs/${name}_report.json"
}

run_proxy touche  data/proxy_touche
run_proxy fiqa    data/proxy_fiqa
run_proxy scifact data/proxy_scifact
run_proxy nfcorpus data/proxy_nfcorpus
run_proxy arguana data/proxy_arguana
run_proxy antique data/proxy_antique

echo
echo "All smoke checks passed."
