#!/bin/bash
# sol_gaudi/build_vllm_gaudi_sif.sh
#
# Fallback: build vllm_gaudi.sif from Habana's public Docker vault via apptainer.
# Run only if gaudi_setup_check.sh reports the container is missing.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env" 2>/dev/null || source "${SCRIPT_DIR}/config.env.example"

mkdir -p "${CONTAINERS_DIR}"

if [[ -f "${VLLM_GAUDI_SIF}" ]]; then
    echo "Container already exists at ${VLLM_GAUDI_SIF} — nothing to do."
    exit 0
fi

if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
    echo "ERROR: apptainer/singularity not on PATH." >&2
    exit 1
fi
BUILDER=$(command -v apptainer || command -v singularity)

echo "Building ${VLLM_GAUDI_SIF} from ${HABANA_VLLM_DOCKER_URI}"
echo "This pulls ~15-20 GB; allow 15-30 minutes on first run."
echo

# Apptainer needs a tmpdir with enough space; /tmp is usually too small
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/${USER}/apptainer_tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/${USER}/apptainer_cache}"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

"${BUILDER}" build "${VLLM_GAUDI_SIF}" "${HABANA_VLLM_DOCKER_URI}"

echo
echo "Built: ${VLLM_GAUDI_SIF}"
du -h "${VLLM_GAUDI_SIF}"
