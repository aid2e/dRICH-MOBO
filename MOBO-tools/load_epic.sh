#!/bin/bash

# taken from epic/install/setup.sh: epic libraries
if [[ "$(uname -s)" = "Darwin" ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        export DYLD_LIBRARY_PATH="${AIDE_HOME}/epic_klm/install/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
else
        export LD_LIBRARY_PATH="${AIDE_HOME}/epic_klm/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
