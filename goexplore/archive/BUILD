package(default_visibility = ["//visibility:public"])

load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "cell",
    srcs = ["cell.py"],
    deps = [],
)

py_library(
    name = "loadable_cell",
    srcs = ["loadable_cell.py"],
    deps = [
        ":cell",
    ],
)
