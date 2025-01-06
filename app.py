from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)
emotion_model = load_model('emotion_model.keras')

def predict_message(pred_text):
  pred_text = np.array([pred_text])
  pred_text = pred_text.astype(np.ndarray)
  outcome = emotion_model.predict(pred_text)
  return str(round(outcome[0][0] * 100, 6)) + "% Sadness " + str(round(outcome[0][1] * 100, 6)) + "% Joy " + str(round(outcome[0][2] * 100, 6)) + "% Love " + str(round(outcome[0][3] * 100, 6)) + "% Anger " + str(round(outcome[0][4] * 100, 6)) + "% Fear " + str(round(outcome[0][5] * 100, 6)) + "% Surprise"

def generate_quote(pred_text):
  pred_text = np.array([pred_text])
  pred_text = pred_text.astype(np.ndarray)
  outcome = emotion_model.predict(pred_text)
  emotion = np.argmax(outcome)
  if emotion == 0:
    return "Your predicted emotion is: sadness. - \"Sad hurts but it’s a healthy feeling.\" – J. K. Rowling"
  elif emotion == 1:
    return "Your predicted emotion is: joy. - \"Don't allow anyone to steal your joy.\" - Paulo Coelho"
  elif emotion == 2:
    return "Your predicted emotion is: love. - \"And now these three remain: faith, hope and love. But the greatest of these is love.\" - 1 Corinthians 13:13"
  elif emotion == 3:
    return "Your predicted emotion is: anger. - \"Holding on to anger is like grasping a hot coal with the intent of throwing it at someone else; you are the one who gets burned.\" - Buddha"
  elif emotion == 4:
    return "Your predicted emotion is: fear. - \"Fear is the brain’s way of saying there is something important for you to overcome.\" - Rachel Huber"
  elif emotion == 5:
    return "Your predicted emotion is: surprise. - \"The best part of the journey is the surprise and wonder along the way.\" - Ken Poirot"

@app.route('/process_input', methods=['POST'])
def process_input():
    # Get input data from the request
    data = request.json
    input_str = data.get('input', '')
    print(f"Received input: {input_str}")  # Debugging line

    # Call your functions
    result_one = predict_message(input_str)
    result_two = generate_quote(input_str)

    # Debugging the results
    print(f"Result one: {result_one}")
    print(f"Result two: {result_two}")

    # Return the results as JSON
    return jsonify({
        'result_one': result_one,
        'result_two': result_two
    })

if __name__ == '__main__':
    app.run(debug=True)