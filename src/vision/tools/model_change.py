import onnxruntime as ort

# 加载你原来的模型
model_path = "/home/bianbu/Emotion_robot/model/vision/enet_b0_8_best_afew.onnx"
output_path = "/home/bianbu/Emotion_robot/model/vision/enet_static.onnx"
model = ort.load(model_path)

# 修改输入维度为固定值 1
model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1

# 有些模型输出也带有动态维度，建议一并修改
model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1

onnx.save(model, output_path)
print(f"静态模型已保存至: {output_path}")