import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("models/adam_v0.3a_350e.keras")

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
