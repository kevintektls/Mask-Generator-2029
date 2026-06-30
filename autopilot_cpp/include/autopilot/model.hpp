#pragma once

#include <memory>
#include <string>
#include <vector>

namespace autopilot {

class OnnxModel {
public:
    explicit OnnxModel(const std::string& onnx_path);
    ~OnnxModel();

    OnnxModel(const OnnxModel&) = delete;
    OnnxModel& operator=(const OnnxModel&) = delete;

    float predict_flat(const std::vector<float>& input);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace autopilot
