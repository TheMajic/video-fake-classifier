from transformers import AutoImageProcessor, AutoModelForVideoClassification
import torch
import cv2
from PIL import Image
import requests
import os

# تحميل الموديل مرة واحدة عند تشغيل السيرفر
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

processor = AutoImageProcessor.from_pretrained(
    "alexgrigore/videomae-base-finetuned-fakeDataset-gesturePhasePleaseWork",
    cache_dir=MODEL_DIR
)

model = AutoModelForVideoClassification.from_pretrained(
    "alexgrigore/videomae-base-finetuned-fakeDataset-gesturePhasePleaseWork",
    cache_dir=MODEL_DIR
)

def classify_video(video_url):
    # تحميل الفيديو من الرابط
    response = requests.get(video_url, stream=True)
    if response.status_code != 200:
        return {"error": "Failed to download video."}

    # حفظ الفيديو مؤقتًا
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as temp_video:
        for chunk in response.iter_content(chunk_size=1024):
            temp_video.write(chunk)

    # استخراج الإطارات من الفيديو
    cap = cv2.VideoCapture(temp_video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        success, frame = cap.read()
    cap.release()
    
    os.remove(temp_video_path)  # حذف الفيديو بعد المعالجة

    if not frames:
        return {"error": "No frames could be extracted from the video."}
    
    # اختيار الإطار الأوسط لتحليله
    mid_frame = frames[len(frames) // 2]
    inputs = processor(images=mid_frame, return_tensors="pt")

    # تشغيل الموديل بدون إعادة تحميله
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
    
    return {"classification": "Real" if predicted_class == 0 else "Fake"}
