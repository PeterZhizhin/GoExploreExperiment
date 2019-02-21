load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "replay_cell_info",
    srcs = ["replay_cell_info.py"],
    deps = [
        "//goexplore/cell:base_returning_info",
    ],
)

py_library(
    name = "replay_policy",
    srcs = ["replay_policy.py"],
    deps = [
        ":replay_cell_info",
        "//goexplore/returning:base_policy",
    ],
)

py_test(
    name = "replay_policy_test",
    srcs = ["replay_policy_test.py"],
    deps = [
        ":replay_policy",
        requirement("gym"),
    ],
)
