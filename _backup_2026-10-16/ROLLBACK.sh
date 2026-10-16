#!/bin/sh
set -eu
# Restore only a manifest copy. Research files and Git history are untouched.
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$#" -ne 1 ]; then
    printf '%s\n' 'Usage: ROLLBACK.sh /absolute/path/to/manifest-copy.json' >&2
    exit 2
fi
if [ "$1" = "$HERE/MODIFIED_FILE.json" ] ||
   { [ -e "$1" ] && [ "$1" -ef "$HERE/MODIFIED_FILE.json" ]; }; then
    printf '%s\n' 'Choose a separate manifest copy; the completed manifest is preserved.' >&2
    exit 2
fi
cp "$HERE/ORIGINAL_FILE.json" "$1"
printf '%s\n' 'RESTORED baseline manifest; research files and Git history unchanged'
