load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "io_bazel_rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    commit = "a558949cce478e537c6474c3bc5848a7d90e42c0",
)

load("@io_bazel_rules_python//python:pip.bzl", "pip_repositories")
pip_repositories()
load("@io_bazel_rules_python//python:pip.bzl", "pip_import")

pip_import(
   name = "pip_deps",
   requirements = "//external:requirements.txt",
)

load("@pip_deps//:requirements.bzl", "pip_install")
pip_install()
