from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from tensorflow.keras.models import load_model

quotes = {
        0: [
            "Even the darkest night will end and the sun will rise.",
            "Tears are words the heart can't express.",
            "The soul would have no rainbow if the eyes had no tears.",
            "Crying is how your heart speaks when your lips can’t explain the pain.",
            "This too shall pass.",
            "Keep your face to the sunshine, and you cannot see the shadow.",
            "Sadness flies away on the wings of time.",
            "Courage does not always roar. Sometimes it is the quiet voice.",
            "Sadness is but a wall between two gardens.",
            "Stars can’t shine without darkness."
        ],
        1: [
            "Happiness is not by chance, but by choice.",
            "Keep your face always toward the sunshine.",
            "The purpose of life is to enjoy every moment.",
            "Joy is the simplest form of gratitude.",
            "Smiles are free, but they are worth a lot.",
            "Let your joy be unconfined.",
            "The only joy in the world is to begin.",
            "Where there is love, there is joy.",
            "A joyful heart is the inevitable result of a heart burning with love.",
            "Choose to be happy."
        ],
        2: [
            "Love is the beauty of the soul.",
            "Love is composed of a single soul inhabiting two bodies.",
            "Where there is love there is life.",
            "Love is not about possession. Love is about appreciation.",
            "To love and be loved is to feel the sun from both sides.",
            "Love is a canvas furnished by nature and embroidered by imagination.",
            "Being deeply loved by someone gives you strength.",
            "Love recognizes no barriers.",
            "The best thing to hold onto in life is each other.",
            "Love is an endless act of forgiveness."
        ],
        3: [
            "Anger is one letter short of danger.",
            "For every minute you are angry you lose sixty seconds of happiness.",
            "Anger doesn’t solve anything. It builds nothing, but it can destroy everything.",
            "He who angers you conquers you.",
            "Holding onto anger is like drinking poison.",
            "Never respond to an angry person with a fiery comeback.",
            "Speak when you are angry and you will make the best speech you will ever regret.",
            "Anger is like a storm rising up from the bottom of your consciousness.",
            "When anger rises, think of the consequences.",
            "Anger is a wind which blows out the lamp of the mind."
        ],
        4: [
            "Do one thing every day that scares you.",
            "Courage is resistance to fear, mastery of fear, not absence of fear.",
            "Fear is only as deep as the mind allows.",
            "Fear defeats more people than any other one thing in the world.",
            "Fear can keep you up all night, but faith makes one fine pillow.",
            "The only thing we have to fear is fear itself.",
            "Courage is not the absence of fear, but the triumph over it.",
            "Fears are nothing more than a state of mind.",
            "Face your fears, don’t run away.",
            "Fear is temporary. Regret is forever."
        ],
        5: [
            "Life is full of surprises, some good, some not so good.",
            "Surprise is the greatest gift which life can grant us.",
            "Expect the unexpected.",
            "Sometimes we’re taken off guard and realize how important something is.",
            "The best things in life are unexpected.",
            "Surprise is the mother of all joy.",
            "Astonishment is the root of philosophy.",
            "The art of life is to live in the present moment.",
            "Be open to the unexpected.",
            "Life is a surprise, enjoy the ride."
        ]
    }

app = Flask(__name__)
CORS(app)
emotion_model = load_model('extracted_model/emotion_model.keras')

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
  if emotion in quotes:
        return random.choice(quotes[emotion])

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

    # Limit TensorFlow to use only necessary GPU memory if running on a GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set a memory limit
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
        except RuntimeError as e:
            print(e)

