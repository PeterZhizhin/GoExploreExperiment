load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "base_archive",
    srcs = ["base_archive.py"],
    deps = [
        "//goexplore/cell:total_cell",
    ],
)

py_library(
    name = "in_memory_archive",
    srcs = ["in_memory_archive.py"],
    deps = [
        ":base_archive",
        "//goexplore/cell:total_cell",
        "//goexplore/cell:base_representation_info",
        requirement("numpy"),
    ],
)

py_test(
    name = "in_memory_archive_test",
    srcs = ["in_memory_archive_test.py"],
    deps = [
        ":in_memory_archive",
        "//goexplore/cell:base_representation_info",
        "//goexplore/cell:base_prioritizing_info",
        "//goexplore/cell:base_utility_info",
        requirement("numpy"),
    ],
)
