FROM --platform=linux/amd64 golang:bookworm AS paddle-amd64
LABEL maintainer="TeCHiScy <741195+TeCHiScy@users.noreply.github.com>"

# note: need >= 16GiB memory to build in docker

ENV BUILD="git \
    sudo \
    build-essential \
    cmake \
    pkg-config \
    python3-yaml \
    python3-jinja2"

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends ${BUILD} && \
    apt-get autoremove -y && apt-get autoclean -y

RUN cd /tmp && git clone -b v2.6.1 --depth=1 https://github.com/PaddlePaddle/Paddle.git && \
    cd Paddle && git checkout v2.6.1 && \
    mkdir build && cd build && \
    sed -i 's/--build ./--build . --parallel 1/' ../cmake/external/gflags.cmake && \
    sed -i 's/--build ./--build . --parallel 1/' ../cmake/external/gloo.cmake && \
    sed -i 's/make/make -j1/' ../cmake/external/xxhash.cmake && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DWITH_PYTHON=OFF \
    -DWITH_GPU=OFF \
    -DWITH_TESTING=OFF \
    -DWITH_MKL=ON \
    -DWITH_MKLDNN=ON \
    -DON_INFER=ON \
    -DWITH_CRYPTO=OFF && \
    ulimit -n 102400 && \
    make -j8 inference_lib_dist

FROM --platform=linux/arm64 golang:bookworm AS paddle-arm64

ENV BUILD="git \
    sudo \
    build-essential \
    cmake \
    pkg-config \
    python3-yaml \
    python3-jinja2"

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends ${BUILD} && \
    apt-get autoremove -y && apt-get autoclean -y

RUN cd /tmp && git clone -b v2.6.1 --depth=1 https://github.com/PaddlePaddle/Paddle.git && \
    cd Paddle && git checkout v2.6.1 && \
    mkdir build && cd build && \
    sed -i 's/--build ./--build . --parallel 1/' ../cmake/external/gflags.cmake && \
    sed -i 's/--build ./--build . --parallel 1/' ../cmake/external/gloo.cmake && \
    sed -i 's/make/make -j1/' ../cmake/external/xxhash.cmake && \
    sed -i '109a set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=class-memaccess")\n' ../CMakeLists.txt && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DWITH_PYTHON=OFF \
    -DWITH_GPU=OFF \
    -DWITH_TESTING=OFF \
    -DON_INFER=ON \
    -DWITH_ARM=ON \
    -DWITH_CRYPTO=OFF && \
    ulimit -n 102400 && \
    make -j8 inference_lib_dist

RUN mkdir -p /tmp/Paddle/build/paddle_inference_c_install_dir/third_party/install/mklml/lib && \
    mkdir -p /tmp/Paddle/build/paddle_inference_install_dir/third_party/install/mkldnn/lib

FROM paddle-${TARGETARCH} AS gocv

ARG OPENCV_VERSION=4.9.0
ENV OPENCV_VERSION=$OPENCV_VERSION

ENV BUILD="git \
    build-essential \
    cmake \
    unzip \
    wget \
    pkg-config"

ENV DEV="libswscale-dev \
    libtbbmalloc2 \
    libtbb-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libtiff-dev"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ${BUILD} ${DEV} && \
    apt-get autoremove -y && apt-get autoclean -y

RUN mkdir /tmp/opencv && \
    cd /tmp/opencv && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip opencv_contrib.zip && \
    mkdir /tmp/opencv/opencv-${OPENCV_VERSION}/build && cd /tmp/opencv/opencv-${OPENCV_VERSION}/build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv/opencv_contrib-${OPENCV_VERSION}/modules \
    -DBUILD_TESTS=NO \
    -DBUILD_PERF_TESTS=NO \
    -DBUILD_EXAMPLES=NO \
    -DBUILD_opencv_apps=NO \
    -DWITH_FFMPEG=NO \
    -DBUILD_JAVA=NO \
    -DBUILD_FAT_JAVA_LIB=NO \
    -DBUILD_opencv_python2=NO \
    -DBUILD_opencv_python3=NO \
    -DWITH_1394=NO \
    -DWITH_ANDROID_MEDIANDK=NO \
    -DWITH_GTK=NO \
    -DOPENCV_GENERATE_PKGCONFIG=YES && \
    make -j8 && \
    make install && \
    cd && rm -rf /tmp/opencv

FROM golang:bookworm AS builder

COPY --from=gocv /usr/local/lib /usr/local/lib
COPY --from=gocv /usr/local/lib/pkgconfig/opencv4.pc /usr/local/lib/pkgconfig/opencv4.pc
COPY --from=gocv /usr/local/include/opencv4/opencv2 /usr/local/include/opencv4/opencv2
COPY --from=gocv /tmp/Paddle/build/paddle_inference_c_install_dir /paddle_inference_c_install_dir
COPY --from=gocv /tmp/Paddle/build/paddle_inference_install_dir /paddle_inference_install_dir

ENV DEV="libswscale-dev \
    libtbbmalloc2 \
    libtbb-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libtiff-dev \
    libomp-dev"

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends ${DEV} && \
    apt-get autoremove -y && apt-get autoclean -y

COPY ./ /build

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/paddle_inference_install_dir/paddle/lib:/paddle_inference_install_dir/third_party/install/mkldnn/lib

RUN cd /build && \
    go mod tidy && \
    ln -s /paddle_inference_c_install_dir ${GOPATH}/pkg/mod/github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi\@v0.0.0-20240301063412-585968367859/paddle_inference_c && \
    go build demo.go

FROM debian:bookworm-slim AS runner

COPY --from=gocv /usr/local/lib /usr/local/lib
COPY --from=gocv /tmp/Paddle/build/paddle_inference_install_dir/paddle/lib /usr/local/lib
COPY --from=gocv /tmp/Paddle/build/paddle_inference_install_dir/third_party/install/mkldnn/lib /usr/local/lib
COPY --from=gocv /tmp/Paddle/build/paddle_inference_c_install_dir/third_party/install/mklml/lib /usr/local/lib
COPY --from=gocv /tmp/Paddle/build/paddle_inference_c_install_dir/paddle/lib/libpaddle_inference_c.so /usr/local/lib/libpaddle_inference_c.so
COPY --from=builder /build/demo /app/demo
COPY ./model /app/model
COPY ./config /app/config

ENV DEV="libjpeg62-turbo \
    libwebp-dev \
    libpng-dev \
    libtiff-dev \
    libgomp1"

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends ${DEV} && \
    apt-get autoremove -y && apt-get autoclean -y

WORKDIR /app