#!/usr/bin/env bash

#set -e

TARGET="$1"

if [ ! -z "$2" ]; then
    CUDA_ARCH="$2"
fi

build_PFPL() {
    echo "installing PFPL..."
    cd PFPL
    make
    cd ..
}

build_zfp() {
    echo "installing cuZFP..."
    cd zfp
    make ZFP_WITH_OPENMP=ON
    cd ..
}

build_SZ3() {
    cd SZ3
    mkdir build 
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)
    make install -j
    cd ../..
}

build_SZo() {
    cd SZo
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd) -DENABLE_AVX2=ON
    make install -j
    cd ../..
}

build_SPERR() {
    cd SPERR
    mkdir build
    cd build
    cmake -DUSE_OMP=ON ..
    make -j
    cd ../..
}

build_tthresh() {
    cd tthresh
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
	make -j
    cd ../..
}
case "$TARGET" in
    PFPL|pfpl)
        build_PFPL
        ;;
    zfp|ZFP)
        build_zfp
        ;;
    SZ3)
	build_SZ3
	;;
    SZo)
	build_SZo
	;;
    SPERR)
	build_SPERR
	;;
    tthresh)
	build_tthresh
	;;
    all)
        build_PFPL
        build_zfp
        build_SZ3
        build_SZo
        build_SPERR
        build_tthresh
        ;;
    *)
        echo "Usage: $0 {PFPL|pfpl|zfp|ZFP|SZ3|SZo|SPERR|tthresh|all}"
#        exit 1
        ;;
esac
