# %% [markdown]
# # Check Koneksi

# %%
import requests
from pprint import PrettyPrinter
 
pp = PrettyPrinter()
pp.pprint(requests.get("http://147.139.204.52:8080/v1/models/spam-email-model").json())

# %% [markdown]
# # Membuat Fungsi prediction request ke model serving

# %%
import os
import tensorflow as tf
import tensorflow_transform as tft
import requests
import json
import base64

def load_model_and_transform_fn(model_url):
    """Load the saved model from TensorFlow Serving"""
    def serve_tf_examples_fn(serialized_tf_examples):
        b64_serialized_examples = base64.b64encode(serialized_tf_examples).decode('utf-8')
        payload = {"instances": [{"examples": {"b64": b64_serialized_examples}}]}
        response = requests.post(model_url, json=payload)
        predictions = json.loads(response.content.decode('utf-8'))['predictions']
        return tf.constant(predictions)

    return serve_tf_examples_fn


url = 'http://147.139.204.52:8080/v1/models/spam-email-model:predict'
serve_tf_examples = load_model_and_transform_fn(url)

def predict(text, serve_tf_examples):
    """Predict function"""
    # Convert text into a serialized tf.Tensor
    example = tf.train.Example()
    example.features.feature[FEATURE_KEY].bytes_list.value.extend([tf.compat.as_bytes(text)])
    serialized_example = example.SerializeToString()

    # Make prediction
    predictions = serve_tf_examples(serialized_example)
    return predictions.numpy()

# %% [markdown]
# # Menguji dan melakukan prediction request ke model serving yang telah dibuat.

# %%
df   = pd.read_csv('dataset\email.csv')
msg = df["Message"][2]

prediction = predict(msg, serve_tf_examples)
print(f'{msg}\n')

if prediction > 0.5:
  print(f"Email classified as spam. Predict Estimate: {prediction}")
else:
  print(f"Email classified as not spam. Predict Estimate: {prediction}")

# %%



