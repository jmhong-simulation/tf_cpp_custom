/*
	TODO:
	- find out how to use gradient descent
	- find out how to use layers
*/

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

int main(int argc, char* argv[])
{
	using namespace tensorflow;

	GraphDef graph_def;

	{
		Scope root = Scope::NewRootScope();
		using namespace ::tensorflow::ops;

		auto A = Const<float>(root.WithOpName("A"), { { 1.0f, 2.0f }, {3.0f, 4.0f } });

		std::cout << typeid(A).name() << std::endl;
		/*std::cout << A.flat<float>().size() << std::endl;*/

		auto x = Const<float>(root.WithOpName("x"), { {1.0f}, {1.0f } });

		auto y = MatMul(root.WithOpName("y"), A, x); // y = Ax

		auto fc1 = ops::Relu(root.WithOpName("relu1"), y);

		TF_CHECK_OK(root.ToGraphDef(&graph_def));
		
		//ops::ApplyGradientDescent(root.WithOpName("y"), )

		//auto cost = ops::Mul(root.WithOpName("cost"), fc1, fc1);

		//auto gd = ops::ApplyGradientDescent(root.WithOpName("GD"), cost, 0.1f);
	}

	Session* session;
	
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	Tensor x_input(DT_FLOAT, TensorShape({2, 1}));
	x_input.flat<float>()(0) = -4.0f;
	x_input.flat<float>()(1) = -5.0f;

	Tensor A_input(DT_FLOAT, TensorShape({ 2, 2 }));
	A_input.flat<float>()(0) = -2.0f;
	A_input.flat<float>()(1) = -3.0f;
	A_input.flat<float>()(2) = -4.0f;
	A_input.flat<float>()(3) = -5.0f;

	std::cout << typeid(x_input.flat<float>()).name() << std::endl;
	std::cout << x_input.flat<float>().size() << std::endl;

	std::cout << typeid(A_input.flat<float>()).name() << std::endl;
	std::cout << A_input.flat<float>().size() << std::endl;
	
	std::vector<std::pair<string, tensorflow::Tensor>> inputs;
	inputs.push_back({ "A", A_input });
	inputs.push_back({ "x", x_input });
		//,
		//{ "x", x_input }		
	//};

	std::vector<tensorflow::Tensor> outputs;

	status = session->Run(inputs, { "relu1" }, {}, &outputs);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}
	std::cout << "Result" << std::endl;
	std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>

	return 0;
}