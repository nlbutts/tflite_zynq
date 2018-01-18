To cross compile TFLITE use the follow command line
bazel build --crosstool_top=third_party/toolchains/cpus/arm:toolchain --cpu=armv7a tensorflow/contrib/lite/examples/simplelite:simplelite