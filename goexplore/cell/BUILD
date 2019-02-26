load("@pip_deps//:requirements.bzl", "requirement")
package(default_visibility = ["//goexplore/archive:__pkg__"])

py_library(
    name = "total_cell",
    srcs = ["total_cell.py"],
    deps = [
        ":base_returning_info",
        ":base_prioritizing_info",
        ":base_representation_info",
        ":base_utility_info",
    ],
)

py_library(
    name = "base_returning_info",
    srcs = ["base_returning_info.py"],
    visibility = [
        "//goexplore/returning:__subpackages__",
        "//goexplore/archive:__pkg__",
    ],
)

py_library(
    name = "base_prioritizing_info",
    srcs = ["base_prioritizing_info.py"],
)

py_library(
    name = "base_representation_info",
    srcs = ["base_representation_info.py"],
)

py_library(
    name = "base_utility_info",
    srcs = ["base_utility_info.py"],
)

py_library(
    name = "reward_prioritization",
    srcs = ["reward_prioritization.py"],
    deps = [
        ":base_prioritizing_info",
    ],
)

py_library(
    name = "base_cell_factory",
    srcs = ["base_cell_factory.py"],
    deps = [
        ":total_cell",
        requirement("gym"),
    ],
)
