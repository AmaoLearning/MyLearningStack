import cv2
import time
from threading import Thread
from Network import NeuralNetwork
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=20):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        #ret = self.stream.set(3,resolution[0])
        #ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

data_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保图像是单通道灰度图
        transforms.Resize((28, 28)),  # 调整图像大小为 28x28
        transforms.RandomRotation(10),
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化图像数据
    ])

videostream = VideoStream(resolution=(480,640),framerate=10).start()
time.sleep(1)

net = NeuralNetwork()
dict = torch.load("model.pth", weights_only=True)
net.load_state_dict(dict["model_state"])
net.double()
net.eval()

icon_map = {
    0: "left",
    1: "pause",
    2: "right",
    3: "stright"
}

threshold = 0.95
default_label = 4

try:
    while True:

        frame = videostream.read()

        frame = cv2.resize(frame, None, fx = 0.25, fy = 0.25, interpolation= cv2.INTER_NEAREST) #采样 160*120

        cv2.imshow("original img", frame)

        # 将BGR转换为HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义蓝色的HSV范围 (你需要根据你的图像调整这些值)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # 创建掩码
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到最大的轮廓 (假设蓝色区域是最大的轮廓)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # 获取边界矩形
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 裁剪图像
            cropped_blue_region = frame[y:y + h, x:x + w]

            # 将裁剪后的蓝色区域转换为PIL Image对象
            cropped_blue_region_img = Image.fromarray(cv2.cvtColor(cropped_blue_region, cv2.COLOR_BGR2RGB))

            # 应用数据转换
            frame = data_transform(cropped_blue_region_img)

            # 将 PyTorch 张量转换回 OpenCV 可以处理的 NumPy 数组
            transformed_frame = frame.numpy().transpose((1, 2, 0)) * 255
            transformed_frame = transformed_frame.astype(dtype=np.uint8)

            cv2.imshow("transformed img", transformed_frame)  # 显示变换后的图像
        
        # 按q键可以退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        with torch.no_grad():
            outputs = net(frame.unsqueeze(0).double())
            probabilities = torch.softmax(outputs, dim=1).numpy()  # 获取概率分布
            predict = torch.argmax(outputs, dim=1).numpy()
            confidence = probabilities[range(len(predict)), predict]  # 获取每个样本最高概率的置信度
        for i, max_label in enumerate(predict):
            print(confidence[i])
            if confidence[i] < threshold:  # 检查置信度是否低于阈值
                max_label = default_label  # 设置为默认标签
            print(f"预测结果：{icon_map.get(max_label, 'Unknown')}")
        time.sleep(0.5)

finally:
    # 确保在程序退出前停止小车
    print("Stopping the vehicle...")
    videostream.stop()
    cv2.destroyAllWindows()