#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
 
using namespace tensorflow;
 
int main(int argc, char* argv[]) {
  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
 
  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), "models/test_graph.pb", &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
 
  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
 
  // Setup inputs and outputs:
 
  // Our graph doesn't require any inputs, since it specifies default values,
  // but we'll change an input to demonstrate.
  Tensor a(DT_FLOAT, TensorShape());
  a.scalar<float>()() = 3.0;
 
  Tensor b(DT_FLOAT, TensorShape());
  b.scalar<float>()() = 2.0;
 
  Tensor x(DT_FLOAT,TensorShape());
  x.scalar<float>()() = 10.0;
 
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "a", a },
    { "b", b },
    { "x", x },
  };
 
  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;
 
  // Run the session, evaluating our "y" operation from the graph
  status = session->Run(inputs, {"y"}, {}, &outputs);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
 
 // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  auto output_y = outputs[0].scalar<float>();
 
  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)
 
  // Print the results
  std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 32>
  std::cout << output_y() << "\n"; // 32
 
  // Free any resources used by the session
  session->Close();
  return 0;
}
