import tensorflow as tf
import os

# Δημιουργούμε ένα απλό μοντέλο με την ίδια αρχιτεκτονική
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(10,))
])
model.compile(optimizer='adam', loss='mse')

# Το σώζουμε με το όνομα που περιμένει το send_to_cloud_test.py
model_path = "aggregated_fog_model.keras"
model.save(model_path)

print(f"Done! Created {model_path}")