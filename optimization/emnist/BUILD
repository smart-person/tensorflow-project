load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "centralized_emnist",
    srcs = ["centralized_emnist.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:centralized_training_loop",
        "//utils/datasets:emnist_dataset",
        "//utils/models:emnist_models",
    ],
)

py_library(
    name = "federated_emnist",
    srcs = ["federated_emnist.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:training_loop",
	"//utils:training_loop_importance",
	"//utils:training_loop_loss",
        "//utils:training_utils",
        "//utils/datasets:emnist_dataset",
        "//utils/models:emnist_models",
    ],
)
