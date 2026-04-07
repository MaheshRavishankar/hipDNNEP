// Copyright (c) 2026, hipDNN EP Authors. All rights reserved.
// Licensed under the MIT License.

#ifndef HIPDNN_EP_TEST_SDPA_TEST_BASE_H_
#define HIPDNN_EP_TEST_SDPA_TEST_BASE_H_

#include "test/common/ep_test_base.h"

// Shared test fixture for SDPA-based tests (MHA and GQA).
// Extends HipDNNTestBase with a convenience RunAndCompare that accepts
// separate Q/K/V tensors and skips when the SDPA engine is unavailable.
class HipDNNSdpaTestBase : public HipDNNTestBase {
 protected:
  void RunAndCompare(
      const char* model_path,
      const std::vector<int64_t>& q_shape,
      const std::vector<float>& q_data,
      const std::vector<int64_t>& k_shape,
      const std::vector<float>& k_data,
      const std::vector<int64_t>& v_shape,
      const std::vector<float>& v_data,
      float tolerance = 1e-3f) {
    ASSERT_TRUE(ep_available_) << "HipDNN EP not available";
    ASSERT_TRUE(IsModelAvailable(model_path))
        << "Model not available at: " << model_path;

    std::vector<std::vector<float>> inputs = {q_data, k_data, v_data};
    std::vector<std::vector<int64_t>> shapes = {q_shape, k_shape, v_shape};
    std::vector<const char*> input_names = {"query", "key", "value"};

    auto cpu_output =
        RunCpu(model_path, inputs, shapes, input_names, "output");

    // The GPU path may throw if the SDPA engine is not available in the
    // current hipDNN build — skip the test rather than failing.
    std::vector<float> gpu_output;
    try {
      gpu_output =
          RunHipDNN(model_path, inputs, shapes, input_names, "output");
    } catch (const std::exception& e) {
      std::string msg = e.what();
      if (msg.find("No engine configurations") != std::string::npos ||
          msg.find("create_execution_plans") != std::string::npos) {
        GTEST_SKIP() << "SDPA engine not available in current hipDNN build: "
                     << msg;
      }
      throw;
    }

    CompareOutputs(cpu_output, gpu_output, tolerance);
  }
};

#endif  // HIPDNN_EP_TEST_SDPA_TEST_BASE_H_
