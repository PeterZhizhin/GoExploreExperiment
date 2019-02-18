load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "save_loadable",
    srcs = ["save_loadable.py"],
    deps = [
        requirement("gym"),
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "save_loadable_test",
    srcs = ["save_loadable_test.py"],
    deps = [
        ":save_loadable",
        requirement("gym"),
    ],
)
