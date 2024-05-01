import tensorflow as tf

LABEL_KEY   = "Category"
FEATURE_KEY = "Message"

# Renaming transformed features
def transformed_name(key):
    return key + "_xf"

# Preprocess input features into transformed features
def preprocessing_fn(inputs):
    """
    inputs:  map from feature keys to raw features
    outputs: map from feature keys to transformed features
    """

    outputs = {}
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    outputs[transformed_name(LABEL_KEY)]   = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
