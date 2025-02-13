import cv2
import numpy as np
import torch

def preprocess_video(video_path, target_size=(224, 224), num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # لا مزيد من الإطارات

        # تغيير الحجم إلى 224x224
        frame = cv2.resize(frame, target_size)

        # تحويل الألوان من BGR إلى RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)
    
    cap.release()

    # التأكد من أن عدد الإطارات يطابق `num_frames`
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))  # تكرار آخر إطار
    elif len(frames) > num_frames:
        frames = frames[:num_frames]  # اقتصاص الزائد

    # تحويل إلى NumPy array ثم إلى Tensor
    frames_array = np.array(frames)  # (num_frames, 224, 224, 3)
    frames_tensor = torch.tensor(frames_array).permute(0, 3, 1, 2)  # (num_frames, 3, 224, 224)

    return frames_tensor

# استخدام الدالة
video_tensor = preprocess_video("Ayman Shoot.mp4")
print(video_tensor.shape)  # يجب أن يكون (16, 3, 224, 224)