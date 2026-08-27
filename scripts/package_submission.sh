#!/usr/bin/env bash
# Usage: bash scripts/package_submission.sh ENTRY_NO
# Example: bash scripts/package_submission.sh 2022CS10999

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

# Copy required assignment files into entry folder
cp "${ROOT_DIR}/conftest.py" "${BUILD_DIR}/"
cp "${ROOT_DIR}/Dockerfile" "${BUILD_DIR}/"
cp "${ROOT_DIR}/README.md" "${BUILD_DIR}/"
cp "${ROOT_DIR}/requirements.txt" "${BUILD_DIR}/"
cp "${ROOT_DIR}/setup.py" "${BUILD_DIR}/"

cp -r "${ROOT_DIR}/docs" "${BUILD_DIR}/"
cp -r "${ROOT_DIR}/harness" "${BUILD_DIR}/"
cp -r "${ROOT_DIR}/scripts" "${BUILD_DIR}/"
cp -r "${ROOT_DIR}/submission" "${BUILD_DIR}/"
cp -r "${ROOT_DIR}/tests" "${BUILD_DIR}/"

# Copy data/toy only (do NOT include data/full)
mkdir -p "${BUILD_DIR}/data"
cp "${ROOT_DIR}/data/README.md" "${BUILD_DIR}/data/"
cp -r "${ROOT_DIR}/data/toy" "${BUILD_DIR}/data/"

# Clean up unwanted binaries, caches, and temp files inside the package
find "${BUILD_DIR}" -name "*.so" -delete
find "${BUILD_DIR}" -name "*.dylib" -delete
find "${BUILD_DIR}" -name "*.pyc" -delete
find "${BUILD_DIR}" -name ".DS_Store" -delete
find "${BUILD_DIR}" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
rm -rf "${BUILD_DIR}/build" "${BUILD_DIR}/.pytest_cache"

# Create the zip archive
(cd /tmp && zip -r "${ZIP_NAME}" "${ENTRY_NO}")
mv "/tmp/${ZIP_NAME}" "${ROOT_DIR}/${ZIP_NAME}"
rm -rf "${BUILD_DIR}"

echo "============================================================"
echo "Submission package created successfully: ${ZIP_NAME}"
echo "Deflates into directory: ./${ENTRY_NO}/"
echo "============================================================"
