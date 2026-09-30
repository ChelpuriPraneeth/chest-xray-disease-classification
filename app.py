from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import os

app = Flask(__name__)

MODEL_PATH = "chest_xray_mobilenetv3.keras"
UPLOAD_FOLDER = os.path.join("static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    filename = None

    if request.method == "POST":

        file = request.files.get("xray")

        if file and file.filename:

            filename = secure_filename(file.filename)
            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            # Load image
            img = tf.keras.utils.load_img(
                filepath,
                target_size=(224, 224)
            )

            # Convert image to array
            img_array = tf.keras.utils.img_to_array(img)

            # Add batch dimension
            img_array = tf.expand_dims(img_array, axis=0)

            # Prediction
            probability = model.predict(
                img_array,
                verbose=0
            )[0][0]

            if probability >= 0.5:
                prediction = "PNEUMONIA"
                confidence = probability * 100
            else:
                prediction = "NORMAL"
                confidence = (1 - probability) * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename
    )


if __name__ == "__main__":
    app.run(debug=True)