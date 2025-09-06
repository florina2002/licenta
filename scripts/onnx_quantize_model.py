from image_calibration_reader import ImageCalibrationDataReader
from vai_q_onnx import quantize_static, VitisQuantFormat, PowerOfTwoMethod
from onnxruntime.quantization import QuantType

reader = ImageCalibrationDataReader(
    image_dir="/home/florina/Downloads/Licenta_scripts/CalibrationDataSet/calibration_images",
    input_name="images",
    image_size=(640, 640)  # match your model input shape
)

quantize_static(
    model_input="best.onnx",
    model_output="best_quant2.onnx",
    calibration_data_reader=reader,
    quant_format=VitisQuantFormat.FixNeuron,
    calibrate_method=PowerOfTwoMethod.NonOverflow,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "AddQDQPairToWeight": True
    }
)

