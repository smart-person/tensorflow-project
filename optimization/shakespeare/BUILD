load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "centralized_shakespeare",
    srcs = ["centralized_shakespeare.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:keras_metrics",
        "//utils:centralized_training_loop",
        "//utils/datasets:shakespeare_dataset",
        "//utils/models:shakespeare_models",
    ],
)

py_library(
    name = "federated_shakespeare",
    srcs = ["federated_shakespeare.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:keras_metrics",
        "//utils:training_loop",
	"//utils:training_loop_importance",
	"//utils:training_loop_loss",
        "//utils:training_utils",
        "//utils/datasets:shakespeare_dataset",
        "//utils/models:shakespeare_models",
    ],
)
py_library(
    name = "federated_shakespeare_random",
    srcs = ["federated_shakespeare_random.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:keras_metrics",
        "//utils:training_loop_random",
        "//utils:training_utils",
        "//utils/datasets:shakespeare_dataset",
        "//utils/models:shakespeare_models",
    ],
)
py_library(
    name = "federated_shakespeare_loss",
    srcs = ["federated_shakespeare_loss.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:keras_metrics",
        "//utils:training_loop_loss",
        "//utils:training_utils",
        "//utils/datasets:shakespeare_dataset",
        "//utils/models:shakespeare_models",
    ],
)