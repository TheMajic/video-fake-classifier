from flask import Flask, request, jsonify
from transformers import AutoImageProcessor
from transformers import AutoModelForVideoClassification
import tempfile
import torch
import cv2
from PIL import Image
import os

app = Flask(__name__)

from process import preprocess_video, load_model

# Load the model and processor
processor, model = load_model()
print("Model loaded successfully.")

# Process the video
video_tensor = preprocess_video("Ayman Shoot.mp4")

# Ensure tensor is in the right format
video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension -> (1, 16, 3, 224, 224)

# Prepare input for the model
inputs = {"pixel_values": video_tensor.float()}  # Convert to float if needed

# Run the model
with torch.no_grad():
    outputs = model(**inputs)

# Get predicted label
logits = outputs.logits
predicted_label = torch.argmax(logits, dim=1).item()

print(f"Predicted Label: {predicted_label}")


def load_model():
    processor = AutoImageProcessor.from_pretrained("alexgrigore/videomae-base-finetuned-fakeDataset-gesturePhasePleaseWork")
    model = AutoModelForVideoClassification.from_pretrained("alexgrigore/videomae-base-finetuned-fakeDataset-gesturePhasePleaseWork")
    return processor, model

processor, model = load_model()

def classify_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        success, frame = cap.read()
    cap.release()
    
    if not frames:
        return {"error": "No frames could be extracted from the video."}
    
    mid_frame = frames[len(frames) // 2]
    inputs = processor(images=mid_frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()
    
    return {"classification": "Real" if predicted_class == 0 else "Fake"}

@app.route("/classify", methods=["POST"])
def classify():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400
    
    video_file = request.files["video"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        video_file.save(temp_video.name)
        video_path = temp_video.name
    
    result = classify_video(video_path)
    os.remove(video_path)  # Clean up temporary file
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
