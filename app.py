# ...existing code...
import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

model = None
_model_path = "model.pkl"
try:
    with open(_model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    # print to console so you see why loading failed
    print(f"Warning: could not load model '{_model_path}': {e}")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # use get(...) and validate
            sepal_length = float(request.form.get("sepal_length", "").strip())
            sepal_width  = float(request.form.get("sepal_width",  "").strip())
            petal_length = float(request.form.get("petal_length", "").strip())
            petal_width  = float(request.form.get("petal_width",  "").strip())

            if model is None:
                return render_template("index.html", prediction_text="Error: model not loaded on server.")

            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            output = model.predict(features)[0]
            return render_template("index.html", prediction_text=f"The flower name is: {output}")

        except ValueError:
            return render_template("index.html", prediction_text="Error: please enter valid numbers.")
        except Exception as e:
            return render_template("index.html", prediction_text="Error: " + str(e))

    # GET -> show form
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
# ...existing code...