#include "autopilot/model.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace autopilot {

struct OnnxModel::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "autopilot_cpp"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::string> input_name_storage;
    std::vector<std::string> output_name_storage;

    explicit Impl(const std::string& onnx_path) {
        session_options.SetIntraOpNumThreads(2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_unique<Ort::Session>(env, onnx_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        size_t n_inputs = session->GetInputCount();
        for (size_t i = 0; i < n_inputs; ++i) {
            auto name = session->GetInputNameAllocated(i, allocator);
            input_name_storage.emplace_back(name.get());
            input_names.push_back(input_name_storage.back().c_str());
        }
        size_t n_outputs = session->GetOutputCount();
        for (size_t i = 0; i < n_outputs; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_name_storage.emplace_back(name.get());
            output_names.push_back(output_name_storage.back().c_str());
        }
    }
};

OnnxModel::OnnxModel(const std::string& onnx_path) : impl_(std::make_unique<Impl>(onnx_path)) {}

OnnxModel::~OnnxModel() = default;

float OnnxModel::predict_flat(const std::vector<float>& input) {
    constexpr int64_t dims[] = {1, 1, 120, 160};
    const size_t expected = static_cast<size_t>(dims[1] * dims[2] * dims[3]);
    if (input.size() != expected) {
        throw std::runtime_error("model input size mismatch");
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(input.data()), input.size(), dims, 4);

    auto outputs = impl_->session->Run(
        Ort::RunOptions{nullptr},
        impl_->input_names.data(), &tensor, 1,
        impl_->output_names.data(), 1);

    float* out = outputs[0].GetTensorMutableData<float>();
    return out[0];
}

}  // namespace autopilot
