from transformers import AutoImageProcessor, AutoModelForVideoClassification
import torch
import cv2
from PIL import Image
import requests
import numpy as np
from io import BytesIO

# تحميل الموديل مرة واحدة عند تشغيل السيرفر
MODEL_DIR = "model"
processor = AutoImageProcessor.from_pretrained(
    "alexgrigore/videomae-base-finetuned-fakeDataset-gesturePhasePleaseWork",
    cache_dir=MODEL_DIR, use_fast=True
)

model = AutoModelForVideoClassification.from_pretrained(
    "alexgrigore/videomae-base-finetuned-fakeDataset-gesturePhasePleaseWork",
    cache_dir=MODEL_DIR
)

def classify_video(video_url):
    # تحميل الفيديو إلى الذاكرة مباشرة
    response = requests.get(video_url, stream=True)
    if response.status_code != 200:
        return {"error": "Failed to download video."}

    video_bytes = BytesIO(response.content)

    # حفظ المحتوى في ملف مؤقت لأن OpenCV لا يدعم BytesIO مباشرة
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as temp_video:
        temp_video.write(video_bytes.getvalue())

    # قراءة الفيديو باستخدام OpenCV
    cap = cv2.VideoCapture(temp_video_path)

    # استخراج عدد الإطارات الكافي
    frames = []
    NUM_FRAMES = 16  # عدد الإطارات المطلوبة للموديل
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < NUM_FRAMES:
        cap.release()
        return {"error": f"Video too short. Required: {NUM_FRAMES} frames, Found: {frame_count} frames"}

    # اختيار إطارات موزعة بالتساوي
    indices = np.linspace(0, frame_count - 1, NUM_FRAMES, dtype=int)
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if not success:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb).convert("RGB"))  # تأكد أن الصورة بها 3 قنوات

    cap.release()

    if len(frames) < NUM_FRAMES:
        return {"error": "Failed to extract enough frames from video."}

    # ✅ استخدم `images=` بدلًا من `video=`
    inputs = processor(images=frames, return_tensors="pt")

    # تشغيل الموديل
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()

    return {"classification": "Real" if predicted_class == 0 else "Fake"}
