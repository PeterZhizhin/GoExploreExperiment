load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "loadable_cell_info",
    srcs = ["loadable_cell_info.py"],
    deps = [
        "//goexplore/cell:base_returning_info",
    ],
)

py_library(
    name = "load_policy",
    srcs = ["load_policy.py"],
    deps = [
        "//goexplore/envs:save_loadable",
        "//goexplore/returning:base_policy",
        ":loadable_cell_info",
    ],
)

py_test(
    name = "load_policy_test",
    srcs = ["load_policy_test.py"],
    deps = [
        "//goexplore/cell:base_returning_info",
        "//goexplore/envs:save_loadable",
        ":load_policy",
        ":loadable_cell_info",
        requirement("gym"),
    ],
)
