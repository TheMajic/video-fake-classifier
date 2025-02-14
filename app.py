from flask import Flask, request, jsonify
from classifier import classify_video

app = Flask(__name__)

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if "video_url" not in data:
        return jsonify({"error": "No video URL provided."}), 400
    
    video_url = data["video_url"]
    result = classify_video(video_url)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
