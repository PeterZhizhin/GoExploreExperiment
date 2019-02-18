load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "base_policy",
    srcs = ["base_policy.py"],
    deps = [
        "//goexplore/archive:cell",
        requirement("gym"),
    ]
)

py_library(
    name = "load_policy",
    srcs = ["load_policy.py"],
    deps = [
        "//goexplore/archive:loadable_cell",
        "//goexplore/envs:save_loadable",
        ":base_policy",
    ]
)

py_test(
    name = "load_policy_test",
    srcs = ["load_policy_test.py"],
    deps = [
        ":load_policy",
        "//goexplore/envs:save_loadable",
        "//goexplore/archive:loadable_cell",
        "//goexplore/archive:cell",
        requirement("gym"),
    ],
)

py_library(
    name = "replay_policy",
    srcs = ["replay_policy.py"],
    deps = [
        "//goexplore/archive:cell",
        ":base_policy",
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
