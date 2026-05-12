import onnxruntime as ort
import spacemit_ort
print('ONNX Runtime:', ort.__version__)
print('可用执行提供程序:', ort.get_available_providers())

'''
import cv2


# 打开摄像头，使用 V4L2 后端
cap = cv2.VideoCapture("/dev/video20", cv2.CAP_V4L2)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 可选：设置像素格式为 MJPG（提高兼容性）
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
# 可选：设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取帧失败")
        break
    # 处理 frame...
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''