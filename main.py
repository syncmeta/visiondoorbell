import cv2
from ultralytics import YOLO, solutions
import pygame #播放音频用

# 音频初始化部分
pygame.mixer.init()
pygame.mixer.music.load("example.mp3")
# 以下代码来自https://docs.ultralytics.com/guides/object-counting/
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)# 0为默认摄像头，可以换成你想要的摄像头，除此之外它还支持rtsp rtmp什么的，实在不行可以用obs的虚拟摄像头来搞
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
# 下面这里以一个人过某条线计数为例，可以自己定义线的位置，也可以定义多个点组成多边形，参考上面链接里的文档
line_points = [(325, 0), (325, 480)]

# 如果要保存输出视频的话，可以用下面这行代码
#video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)
count = 0;
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    #这里只对人计数
    classes_to_count = [0]  # 0是人
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)
    im0 = counter.start_counting(im0, tracks)
    # 写入视频用下面这行
    #video_writer.write(im0)
    print(counter.out_counts) # 输出计数，从左到右过线计入out_counts，从右到左过线计入in_counts，下面同理
    if(counter.out_counts>count):
        # 有人进入时执行以下代码，这里以播放音频为例
        pygame.mixer.music.play()
        # 等待音频播放完毕
        while pygame.mixer.music.get_busy():
            continue
        count = counter.out_counts

cap.release()
#video_writer.release()
cv2.destroyAllWindows()