#!/bin/bash

if [ "$#" != 2 ]; then
    echo "Usage: $0 [EICrecon directory] [install directory]"
    exit 1
fi


EPICDIR=$1
echo $EPICDIR
INSTALLDIR=$2
echo $INSTALLDIR
cd $EPICDIR
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=$INSTALLDIR
cmake --build build
cmake --install build
source $INSTALLDIR/bin/eicrecon-this.sh
