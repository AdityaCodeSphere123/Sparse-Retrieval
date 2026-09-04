#!/usr/bin/env bash
# Usage: bash scripts/package_submission.sh ENTRY_NO
# Example: bash scripts/package_submission.sh 2022CS10999
#
# Packs the Moodle zip so that it inflates to:
#   ENTRY_NO/
#     assignment1.tex
#     conftest.py
#     data/README.md
#     data/toy/
#     Dockerfile
#     docs/
#     harness/
#     README.md
#     report.tex
#     requirements.txt
#     runs/
#     scripts/
#     submission/
#       setup.py          (pybind11: sources under submission/code/)
#       retrieve.py
#       code/
#         ir_config.h engine.h
#         analysis.{h,cpp} codecs.{h,cpp} corpus_io.{h,cpp} bindings.cpp
#     tests/
#

set -e

if [ -z "$1" ]; then
  echo "Error: Missing entry number argument."
  echo "Usage: bash scripts/package_submission.sh <ENTRY_NUMBER>"
  echo "Example: bash scripts/package_submission.sh 2022CS10999"
  exit 1
fi

ENTRY_NO=$(echo "$1" | tr '[:lower:]' '[:upper:]')
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="/tmp/${ENTRY_NO}"
ZIP_NAME="${ENTRY_NO}.zip"

echo "Packaging submission for Entry No: ${ENTRY_NO}..."

rm -rf "${BUILD_DIR}" "${ROOT_DIR}/${ZIP_NAME}"
mkdir -p "${BUILD_DIR}"

cp "${ROOT_DIR}/assignment1.tex" "${BUILD_DIR}/"
cp "${ROOT_DIR}/conftest.py" "${BUILD_DIR}/"
cp "${ROOT_DIR}/Dockerfile" "${BUILD_DIR}/"
cp "${ROOT_DIR}/README.md" "${BUILD_DIR}/"
cp "${ROOT_DIR}/requirements.txt" "${BUILD_DIR}/"
if [ -f "${ROOT_DIR}/report.tex" ]; then
  cp "${ROOT_DIR}/report.tex" "${BUILD_DIR}/"
fi
if [ -f "${ROOT_DIR}/report.pdf" ]; then
  cp "${ROOT_DIR}/report.pdf" "${BUILD_DIR}/"
fi

cp -r "${ROOT_DIR}/docs" "${BUILD_DIR}/"
cp -r "${ROOT_DIR}/harness" "${BUILD_DIR}/"
cp -r "${ROOT_DIR}/submission" "${BUILD_DIR}/"
cp -r "${ROOT_DIR}/tests" "${BUILD_DIR}/"

# Starter scripts only (no local sweeps / proxy builders).
mkdir -p "${BUILD_DIR}/scripts"
for f in smoke_test.sh download_full_corpus.py error_analysis.py; do
  cp "${ROOT_DIR}/scripts/${f}" "${BUILD_DIR}/scripts/"
done

mkdir -p "${BUILD_DIR}/data"
cp "${ROOT_DIR}/data/README.md" "${BUILD_DIR}/data/"
cp -r "${ROOT_DIR}/data/toy" "${BUILD_DIR}/data/"

mkdir -p "${BUILD_DIR}/runs"
if [ -d "${ROOT_DIR}/runs" ]; then
  # Starter-tracked run artifacts only; skip local sweep dumps.
  for f in dev_run.trec dev_report.json; do
    if [ -f "${ROOT_DIR}/runs/${f}" ]; then
      cp "${ROOT_DIR}/runs/${f}" "${BUILD_DIR}/runs/"
    fi
  done
fi

# Compiled extension + local junk. The grader rebuilds from setup.py + code/.
find "${BUILD_DIR}" -name "*.so" -delete
find "${BUILD_DIR}" -name "*.dylib" -delete
find "${BUILD_DIR}" -name "*.o" -delete
find "${BUILD_DIR}" -name "*.pyc" -delete
find "${BUILD_DIR}" -name ".DS_Store" -delete
find "${BUILD_DIR}" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${BUILD_DIR}" -name "_extract_*.txt" -delete
rm -rf "${BUILD_DIR}/build" "${BUILD_DIR}/.pytest_cache"
rm -rf "${BUILD_DIR}/submission/build" "${BUILD_DIR}/submission/code/build"
# Old layout name; do not ship a second copy of the sources.
rm -rf "${BUILD_DIR}/submission/cpp"
# Legacy single-file engine must not ship; setup.py compiles code/*.cpp.
rm -f "${BUILD_DIR}/submission/cpp_engine.cpp"

CODE_DIR="${BUILD_DIR}/submission/code"
REQUIRED_SRC=(
  ir_config.h
  engine.h
  analysis.h analysis.cpp
  codecs.h codecs.cpp
  corpus_io.h corpus_io.cpp
  bindings.cpp
)
missing=0
for f in "${REQUIRED_SRC[@]}"; do
  if [ ! -f "${CODE_DIR}/${f}" ]; then
    echo "Error: missing submission/code/${f} (needed by submission/setup.py)"
    missing=1
  fi
done
if [ ! -f "${BUILD_DIR}/submission/setup.py" ]; then
  echo "Error: missing submission/setup.py"
  missing=1
fi
if [ "${missing}" -ne 0 ]; then
  exit 1
fi

# Guard: never ship the assignment PDF or the full corpus.
rm -f "${BUILD_DIR}/assignment1.pdf"
rm -rf "${BUILD_DIR}/data/full"

(cd /tmp && zip -r "${ZIP_NAME}" "${ENTRY_NO}")
mv "/tmp/${ZIP_NAME}" "${ROOT_DIR}/${ZIP_NAME}"
rm -rf "${BUILD_DIR}"

echo "============================================================"
echo "Submission package created successfully: ${ZIP_NAME}"
echo "Deflates into directory: ./${ENTRY_NO}/"
echo "============================================================"
