# -*- Python -*-
# Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
# Licensed under the MIT License.

import os
import lit.formats
import lit.util

config.name = "HipDNN Torch-MLIR"
config.test_format = lit.formats.ShTest(not lit.util.which("bash"))

config.suffixes = [".test", ".py"]
config.excludes = ["lit.cfg.py", "lit.site.cfg.py"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.hipdnn_obj_root, "test", "lit")

# Substitutions
config.substitutions.append(("%FileCheck", config.filecheck_path))
config.substitutions.append(("%hipdnn-onnx-to-torch-mlir", config.hipdnn_onnx_to_torch_mlir))
config.substitutions.append(("%S", config.test_source_root))
config.substitutions.append(("%B", config.test_exec_root))

# Environment
config.environment["PATH"] = os.pathsep.join([
    os.path.dirname(config.hipdnn_onnx_to_torch_mlir),
    config.environment.get("PATH", "")
])
