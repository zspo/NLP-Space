#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <chrono>

#define USE_CPU // Chnage USE_CPU to USE_CUDA

#ifdef USE_CUDA
#include "cuda_provider_factory.h"
#endif  // CUDA GPU Enabled

// export LD_LIBRARY_PATH=${}/onnxruntime-linux-x64-1.9.0/lib:$LD_LIBRARY_PATH
// export LD_LIBRARY_PATH=${}/onnxruntime-linux-x64-gpu-1.9.0/lib:$LD_LIBRARY_PATH
// g++ a.cpp -o a ${}/onnxruntime-linux-x64-1.9.0/lib/libonnxruntime.so.1.9.0 -I ${}/onnxruntime-linux-x64-1.9.0/include/ -std=c++11
// g++ a.cpp -o a ${}/onnxruntime-linux-x64-gpu-1.9.0/lib/libonnxruntime.so.1.9.0 -I ${}/onnxruntime-linux-x64-gpu-1.9.0/include/ -std=c++11


int main() {
    int round = 1000;
    std::cout << round << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef USE_CUDA
	Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif  // CUDA GPU Enabled

    const char* model_path = "./mybert.onnx";
    Ort::Session session(env, model_path, session_options);

    //// print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;
    char* output_name = session.GetOutputName(0, allocator);
    std::cout << output_name << std::endl;

    std::vector<const char*> input_node_names = {"input_ids", "token_type_ids", "attention_mask"};
    std::vector<const char*> output_node_names = {"logits"};

    // input_ids
    std::vector<int64_t> input_ids_dims = {1, 82};
    size_t input_ids_size = 1 * 82; 
    auto memory_info_1 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // std::vector<long> input_ids_value = {101, 1037, 3899, 2003, 2770, 2006, 3509,  102};
    // std::vector<long> input_ids_value = {101 ,1037 ,3899 ,2003 ,2770 ,2006 ,3509 ,102 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0};
    std::vector<int64_t> input_ids_value = {101, 11724,  8762, 12126,  8168,   150,  8179, 10006, 10600, 10168, 10614,  9738,  9107,  8847,  9479, 11839,  8521,  8361, 10168, 11014, 8217,  9568,  9116,  8809,  9470, 12183,  8877,  9145, 11233,  9428,8134, 11104, 12729,  8913, 11057,  9202,  9374,  8139,  9392,  8154,8231,  8606, 12126,  8168,   150,  8179, 10006, 10600,  8346,  8998,9019, 11685,  8797,  9749,  8675, 10447,  8328, 11399,  9796, 11588,8180, 10091,  9786,  8165, 11399, 10537, 10367, 10242,  8178, 10484,12619, 12465, 10361,  8178,  8343,  9531,  8171, 12280,  8317,  9194,8736,   102};
    Ort::Value input_ids = Ort::Value::CreateTensor<int64_t>(memory_info_1, input_ids_value.data(), input_ids_size, input_ids_dims.data(), 2);
    assert(input_ids.IsTensor());
    // token_type_ids
    std::vector<int64_t> token_type_ids_dims = {1, 82};
    size_t token_type_ids_size = 1 * 82; 
    auto memory_info_2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // std::vector<long> token_type_ids_value = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int64_t> token_type_ids_value;
    for (int i = 0; i < 82; ++ i) {
        token_type_ids_value.push_back(0);
    }
    Ort::Value token_type_ids = Ort::Value::CreateTensor<int64_t>(memory_info_2, token_type_ids_value.data(), token_type_ids_size, token_type_ids_dims.data(), 2);
    assert(token_type_ids.IsTensor());
    // attention_mask
    std::vector<int64_t> attention_mask_dims = {1, 82};
    size_t attention_mask_size = 1 * 82; 
    auto memory_info_3 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // std::vector<long> attention_mask_value = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int64_t> attention_mask_value;
    for (int i = 0; i < 82; ++ i) {
        attention_mask_value.push_back(1);
    }
    Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(memory_info_3, attention_mask_value.data(), attention_mask_size, attention_mask_dims.data(), 2);
    assert(attention_mask.IsTensor());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_ids));
    ort_inputs.push_back(std::move(token_type_ids));
    ort_inputs.push_back(std::move(attention_mask));

    // test time
    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < round; ++ i) {
        session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("time cost: %.3f seconds\n", elapsed.count() * 1e-9);
    // auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 2);
  
    // Get pointer to output tensor float values
    // auto type_info = output_tensors[1].GetTensorTypeAndShapeInfo();
    // for (auto x: type_info.GetShape())
    //     std::cout << "shape " << x << std::endl;
    // std::cout << "len " << type_info.GetElementCount() << std::endl;
    // float* sequence = output_tensors[0].GetTensorMutableData<float>();
    // float* pooled = output_tensors[1].GetTensorMutableData<float>();
    // for (size_t i = 0; i != type_info.GetElementCount(); ++ i) {
    //     std::cout << pooled[i] << " ";
    // }
    // std::cout << pooled[0] << std::endl;
    

    return 0;
}