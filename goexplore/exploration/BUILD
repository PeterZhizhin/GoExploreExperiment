load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "base_policy",
    srcs = ["base_policy.py"],
    deps = [
        requirement("gym"),
    ]
)

py_library(
    name = "random_action_policy",
    srcs = ["random_action_policy.py"],
    deps = [
        "base_policy",
    ]
)

py_test(
    name = "random_action_policy_test",
    srcs = ["random_action_policy_test.py"],
    deps = [
        "random_action_policy",
        requirement("gym"),
    ],
)
