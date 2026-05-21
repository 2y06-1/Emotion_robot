import onnx

m = onnx.load("/home/bianbu/Emotion_robot/model/asr/model.int8.onnx")

for i in m.graph.initializer[:5]:
    print(i.data_type)