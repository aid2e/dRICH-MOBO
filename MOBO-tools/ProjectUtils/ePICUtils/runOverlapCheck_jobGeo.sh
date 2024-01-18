#!/bin/bash

checkOverlaps -c ${DETECTOR_PATH}/${DETECTOR_CONFIG}_$1.xml > $AIDE_HOME/log/overlaps/overlap_log_$1.txt 2>&1
