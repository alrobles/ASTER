#!/usr/bin/env bash
set -euo pipefail

RECIPE="${1:-}"
OUT="${2:-}"

case "$RECIPE" in
    rocm|cuda) ;;
    *)
        echo "usage: $0 rocm|cuda [output.sif]" >&2
        exit 2
        ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF="${HERE}/caster-${RECIPE}.def"
OUT="${OUT:-${HERE}/caster-${RECIPE}.sif}"

command -v apptainer >/dev/null 2>&1 || {
    echo "apptainer not found" >&2
    exit 1
}

export APPTAINER_CACHEDIR="${
    APPTAINER_CACHEDIR:-${TMPDIR:-/tmp}/apptainer-cache-$(id -u)
}"
mkdir -p "$APPTAINER_CACHEDIR"

echo "recipe=$DEF"
echo "output=$OUT"
echo "apptainer=$(apptainer --version)"
echo "cache=$APPTAINER_CACHEDIR"

build_image() {
    local mode="$1"
    shift
    echo "attempt=$mode"
    apptainer build "$@" --force "$OUT" "$DEF"
}

if build_image fakeroot --fakeroot; then
    exit 0
fi

if build_image plain; then
    exit 0
fi

echo "apptainer build failed with fakeroot and plain modes" >&2
exit 1
