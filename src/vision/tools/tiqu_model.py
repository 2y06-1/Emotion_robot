import onnx
import numpy as np
from onnx import numpy_helper

# 加载你之前生成的 fixed1 模型
model_path = "/home/bianbu/Emotion_robot/model/vision/enet_fixed1.onnx"
model = onnx.load(model_path)

# 1. 查找 Gemm_532 节点
gemm_node = None
for node in model.graph.node:
    if node.name == "Gemm_532" or (len(node.output) > 0 and node.output[0] == "output"):
        gemm_node = node
        break

if not gemm_node:
    print("未找到 Gemm 节点，请尝试检查模型节点名称")
    exit()

# 2. 提取权重 (W) 和 偏置 (B)
# Gemm 的输入顺序通常是 [X, W, B]
weight_name = gemm_node.input[1]
bias_name = gemm_node.input[2]
input_to_gemm = gemm_node.input[0] # 进入 Gemm 之前的那个张量

weights = {}
for initializer in model.graph.initializer:
    if initializer.name == weight_name:
        weights['W'] = numpy_helper.to_array(initializer)
    if initializer.name == bias_name:
        weights['B'] = numpy_helper.to_array(initializer)

# 保存权重供后续 Python 使用
np.savez("/home/bianbu/Emotion_robot/model/vision/gemm_weights.npz", W=weights['W'], B=weights['B'])
print("--- 权重已保存至 gemm_weights.npz ---")

# 3. 截断模型：移除 Gemm 节点，将输入设为新输出
# 找到原有的输出并移除
while len(model.graph.output) > 0:
    model.graph.output.pop()

# 添加新的输出 (Gemm 的输入张量)
# 这里的维度应该是 [1, 1280]
new_output = onnx.helper.make_tensor_value_info(input_to_gemm, onnx.TensorProto.FLOAT, [1, 1280])
model.graph.output.append(new_output)

# 从图中删除 Gemm 节点
model.graph.node.remove(gemm_node)

# 保存新模型
onnx.save(model, "/home/bianbu/Emotion_robot/model/vision/enet_backbone.onnx")
print("--- 无尾模型已保存至 enet_backbone.onnx ---")