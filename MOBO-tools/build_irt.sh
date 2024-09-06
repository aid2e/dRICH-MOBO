#!/bin/bash

if [ "$#" != 2 ]; then
    echo "Usage: $0 [irt directory] [install directory]"
    exit 1
fi


IRTDIR=$1
echo $IRTDIR
INSTALLDIR=$2
echo $INSTALLDIR
cd $IRTDIR
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTALLDIR -DEVALUATION=NO ../
make -j2 install
