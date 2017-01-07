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
	/*status = ReadBinaryProto(Env::Default(), "D:/tftest/graph.pb", &graph_def);*/
	status = ReadTextProto(Env::Default(), "D:/tftest/xor_4batch_graph.pb", &graph_def);

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

	//graph_def.add_node();

	// Setup inputs and outputs:

	// Our graph doesn't require any inputs, since it specifies default values,
	// but we'll change an input to demonstrate.
	Tensor x(DT_FLOAT, TensorShape({ 4, 2 })); // x_input = tf.placeholder(tf.float32, shape=(4, 2), name="x-input")
	x.flat<float>()(0) = 0;	x.flat<float>()(1) = 0;
	x.flat<float>()(2) = 0;	x.flat<float>()(3) = 1;
	x.flat<float>()(4) = 1; x.flat<float>()(5) = 0;
	x.flat<float>()(6) = 1;	x.flat<float>()(7) = 1;

	Tensor y(DT_FLOAT, TensorShape({ 4, 1 })); // y_input = tf.placeholder(tf.float32, shape=[4,1], name="y-input")
	y.flat<float>()(0) = 0;
	y.flat<float>()(1) = 1;
	y.flat<float>()(2) = 1;
	y.flat<float>()(3) = 0;

	std::vector<std::pair<string, tensorflow::Tensor>> inputs;
	inputs.push_back({ "x-input", x });
	inputs.push_back({ "y-input", y });

	// The session will initialize the outputs
	std::vector<tensorflow::Tensor> outputs;

	// Run the session, evaluating our "c" operation from the graph
	status = session->Run(inputs, { "y-output" }, { "init" }, &outputs);

	for (int i = 0; i < 100000; i++)
	{

		status = session->Run(inputs, { "y-output" }, { "GradientDescent" }, &outputs);
		if (!status.ok()) {
			std::cout << status.ToString() << "\n";
			return 1;
		}

		if (i % 10000 == 0)
		{
			// Grab the first output (we only evaluated one graph node: "c")
			// and convert the node to a scalar representation.
			auto output_c = outputs[0].flat<float>();

			// (There are similar methods for vectors and matrices here:
			// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

			// Print the results
			//std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
			std::cout << output_c(0) << " " << output_c(1) << " " << output_c(2) << " " << output_c(3) << "\n"; // 30
		}
	}

    // Free any resources used by the session
	session->Close();
	return 0;
}