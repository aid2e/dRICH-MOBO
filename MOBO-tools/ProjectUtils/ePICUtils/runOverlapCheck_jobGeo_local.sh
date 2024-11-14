#!/bin/bash

# checkOverlaps -c ${DETECTOR_PATH}/${DETECTOR_CONFIG}_$1.xml > $AIDE_WORKDIR/log/overlaps/overlap_log_$1.txt 2>&1
# echo ls -l ${AIDE_WORKDIR}/share/epic/
# ls -l ${AIDE_WORKDIR}/share/epic/

# echo "env"
# env

# echo "cat ${AIDE_WORKDIR}/share/epic/${DETECTOR_CONFIG}_$1.xml"
# cat ${AIDE_WORKDIR}/share/epic/${DETECTOR_CONFIG}_$1.xml

echo "checkOverlaps -c ${AIDE_WORKDIR}/share/epic/${DETECTOR_CONFIG}_$1.xml > $AIDE_WORKDIR/log/overlaps/overlap_log_$1.txt 2>&1"

checkOverlaps -c ${AIDE_WORKDIR}/share/epic/${DETECTOR_CONFIG}_$1.xml > $AIDE_WORKDIR/log/overlaps/overlap_log_$1.txt 2>&1

# echo "cat $AIDE_WORKDIR/log/overlaps/overlap_log_$1.txt"
# cat $AIDE_WORKDIR/log/overlaps/overlap_log_$1.txt
