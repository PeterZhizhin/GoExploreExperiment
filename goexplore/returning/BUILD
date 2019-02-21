load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "base_policy",
    srcs = ["base_policy.py"],
    deps = [
        "//goexplore/cell:base_returning_info",
        requirement("gym"),
    ],
    visibility = ["//visibility:public"],
)

