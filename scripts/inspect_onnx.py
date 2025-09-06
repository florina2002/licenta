import onnx

model = onnx.load("best.onnx")
input_names = [inp.name for inp in model.graph.input]
print("Model Input Names:", input_names)

