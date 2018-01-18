package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/contrib/lite:build_def.bzl", "tflite_copts", "gen_selected_ops")

config_setting(
    name = "mips",
    values = {
        "cpu": "mips",
    },
)

config_setting(
    name = "mips64",
    values = {
        "cpu": "mips64",
    },
)

cc_binary(
    name = "simplelite",
    defines = ["USE_NEON"],
    copts = ["-ggdb"],
    srcs = ["simplelite.cc"],
    linkopts = ["-lm -lopencv_core -lopencv_imgcodecs -lopencv_imgproc"],
    deps = [
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ],
)
