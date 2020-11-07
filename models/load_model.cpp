/*************************************************************************
    > File Name: load_model.cpp
    > Author: ce39906
    > Mail: ce39906@163.com
    > Created Time: 2018-09-08 08:28:51
 ************************************************************************/
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <iostream>
#include <string>

const static std::string kModelPath = "models/test_model.pb";

int main() {
    using namespace tensorflow;
//    auto session = NewSession(SessionOptions());
    Session* session;
    Status status;
    status = NewSession(SessionOptions(), &session);

    if ( !status.ok() ) {
        std::cout << status.ToString() << "\n";
        return 1;
    }
    else
    {
        std::cout << "Tensorflow session create success.\n";
    }

    // Read in the protobuf graph we exported
    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), kModelPath, &graph_def);
    if (!status.ok())
    {
        std::cerr << "Error reading graph definition from " << kModelPath
            << ": " << status.ToString();
        return -1;
    }
    else
    {
        std::cout << "Read graph def success.\n";
    }
    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok())
    {
        std::cerr << "Error creating graph: " << status.ToString();
        return -1;
    }
    else
    {
        std::cout << "Create graph success.\n";
    }
    // Set model input
    Tensor hello(DT_STRING, TensorShape());
    hello.scalar<string>()() = "hello";
    Tensor tensorflow(DT_STRING, TensorShape());
    tensorflow.scalar<string>()() = "tensorflow";
    // Apply the loaded model
    std::vector<std::pair<string, tensorflow::Tensor>> inputs =
    {
        { "a", hello },
        { "b", tensorflow },
    }; // input
    std::vector<tensorflow::Tensor> outputs; // output
    status = session->Run(inputs, {"result"}, {}, &outputs);
    if (!status.ok())
    {
        std::cerr << status.ToString() << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Run session successfully" << std::endl;
    }
    // Output the result
    const auto result = outputs[0].scalar<string>()();
    std::cout << "Result value: " << result << std::endl;
    status = session->Close();
    if (!status.ok())
    {
        std::cerr << "Session closed success";
        return -1;
    }
    return 0;
}
