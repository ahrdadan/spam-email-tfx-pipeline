# %%
#!pip install -r requirements.txt

# %% [markdown]
# # Import Package

# %%
import os
import pandas as pd

import tensorflow as tf
import tensorflow_model_analysis as tfma

from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator
from tfx.components import Transform, Trainer, Tuner, Evaluator, Pusher
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

# %% [markdown]
# # Load Data

# %%
os.environ['KAGGLE_USERNAME'] = 'ahrdadan'
os.environ['KAGGLE_KEY']      = '7eb307ba5bd108dae806763473bd568c'

# %%
!kaggle datasets download -d ashfakyeafi/spam-email-classification -f email.csv

# %%
DATA_PATH = 'dataset'

DATASET_NAME = 'email.csv'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

df = pd.read_csv(DATASET_NAME)

df['Category'] = df['Category'].map({'spam': 1, 'ham': 0})
df['Category'] = df['Category'].fillna(0)
df['Category'] = df['Category'].astype(int)

df.to_csv(os.path.join(DATA_PATH, DATASET_NAME), index=False)
df = pd.read_csv(f'{DATA_PATH}/{DATASET_NAME}')

os.remove(DATASET_NAME)

df.head()

# %% [markdown]
# # Pipeline
# ---

# %% [markdown]
# 
# Initial interactive context and directory

# %%
PIPELINE_NAME = 'dhadhan-pipeline'
SCHEMA_PIPELINE_NAME = 'spam-email-tfdv-schema'

PIPELINE_ROOT = os.path.join(PIPELINE_NAME)
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

DATA_ROOT = DATA_PATH

context = InteractiveContext(pipeline_root=PIPELINE_ROOT)

# %% [markdown]
# ## Data Ingestion

# %%
output = example_gen_pb2.Output(
    split_config = example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=3),
        example_gen_pb2.SplitConfig.Split(name='eval',  hash_buckets=1)
    ])
)

example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output)

# %%
context.run(example_gen)

# %% [markdown]
# ## Data Validation

# %%
statistics_gen = StatisticsGen(
    examples = example_gen.outputs['examples']
)

context.run(statistics_gen)

# %% [markdown]
# ### Buat data summary dengan statistic

# %%
context.show(statistics_gen.outputs["statistics"])

# %% [markdown]
# ### Membuat data schema

# %%
schema_gen = SchemaGen(
    statistics = statistics_gen.outputs['statistics']
)

context.run(schema_gen)

# %%
context.show(schema_gen.outputs['schema'])

# %% [markdown]
# ### Check Anomali pada dataset

# %%
example_validator = ExampleValidator(
    statistics = statistics_gen.outputs['statistics'],
    schema     = schema_gen.outputs['schema']
)

context.run(example_validator)

# %%
context.show(example_validator.outputs['anomalies'])

# %% [markdown]
# ## Data Preprocessing
# Data preprocessing dengan module transform

# %%
TRANSFORM_MODULE_FILE = 'spam_email_transform.py'

# %%

%%writefile {TRANSFORM_MODULE_FILE}
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

# %%

transform = Transform(
    examples    = example_gen.outputs['examples'],
    schema      = schema_gen.outputs['schema'],
    module_file = os.path.abspath(TRANSFORM_MODULE_FILE)
)

context.run(transform)

# %% [markdown]
# ## Tuning Model

# %%
TUNER_MODULE_FILE = 'spam_email_tuner.py'

# %%
%%writefile {TUNER_MODULE_FILE}
import os
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner as kt
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from keras_tuner.engine import base_tuner
from typing import NamedTuple, Dict, Text, Any

LABEL_KEY   = "Category"
FEATURE_KEY = "Message"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64) -> tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""

    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern = file_pattern,
        batch_size   = batch_size,
        features     = transform_feature_spec,
        reader       = gzip_reader_fn,
        num_epochs   = num_epochs,
        label_key    = transformed_name(LABEL_KEY)
    )

    return dataset

# Vocabulary size and number of words in a sequence.
VOCAB_SIZE      = 8000
SEQUENCE_LENGTH = 100

vectorize_layer = layers.TextVectorization(
    standardize            = 'lower_and_strip_punctuation',
    max_tokens             = VOCAB_SIZE,
    output_mode            = 'int',
    output_sequence_length = SEQUENCE_LENGTH
)

def model_builder(hp):
    """Build keras tuner model"""
    embedding_dim = hp.Int('embedding_dim', min_value=16, max_value=128, step=16)
    lstm_units    = hp.Int('lstm_units', min_value=16, max_value=128, step=16)
    num_layers    = hp.Choice('num_layers', values=[1, 2, 3])
    dense_units   = hp.Int('dense_units', min_value=16, max_value=128, step=16)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])

    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)

    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, embedding_dim, name='embedding')(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
    for _ in range(num_layers):
        x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    model.compile(
        loss      = tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer = tf.keras.optimizers.Adam(learning_rate),
        metrics   = [tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model

TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', base_tuner.BaseTuner),
    ('fit_kwargs', Dict[Text, Any]),
])

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor  = 'val_binary_accuracy',
    mode     = 'max',
    verbose  = 1,
    patience = 2,
    min_delta= 0,
    baseline = 0.9,
    restore_best_weights =True
)

def tuner_fn(fn_args: FnArgs) -> None:
    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, 10)
    val_set   = input_fn(fn_args.eval_files[0],  tf_transform_output, 10)

    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)
        ]]
    )

    # Build the model tuner
    model_tuner = kt.Hyperband(
        hypermodel   = lambda hp: model_builder(hp),
        objective    = kt.Objective('val_binary_accuracy', direction='max'),
        max_epochs   = 5,
        factor       = 3,
        directory    = 'dhadhan-pipeline',
        project_name = 'spam_email_tuner',
    )

    model_tuner.oracle.max_trials = 3

    return TunerFnResult(
        tuner      = model_tuner,
        fit_kwargs = {
            'callbacks'        : [early_stop_callback],
            'x'                : train_set,
            'validation_data'  : val_set,
            'steps_per_epoch'  : fn_args.train_steps,
            'validation_steps' : fn_args.eval_steps
        }
    )

# %%
tuner = Tuner(
    module_file     = os.path.abspath(TUNER_MODULE_FILE),
    examples        = transform.outputs['transformed_examples'],
    transform_graph = transform.outputs['transform_graph'],
    schema          = schema_gen.outputs['schema'],
    train_args      = trainer_pb2.TrainArgs(splits=['train']),
    eval_args       = trainer_pb2.EvalArgs(splits=['eval'])
)


context.run(tuner)

# %% [markdown]
# ## Model Training

# %%
TRAINER_MODULE_FILE = "spam_email_trainer.py"

# %%
%%writefile {TRAINER_MODULE_FILE}
import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY   = "Category"
FEATURE_KEY = "Message"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64) -> tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""

    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    # Create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern = file_pattern,
        batch_size   = batch_size,
        features     = transform_feature_spec,
        reader       = gzip_reader_fn,
        num_epochs   = num_epochs,
        label_key    = transformed_name(LABEL_KEY)
    )

    return dataset

# Vocabulary size and number of words in a sequence
VOCAB_SIZE      = 8000
SEQUENCE_LENGTH = 100
embedding_dim   = 16

vectorize_layer = layers.TextVectorization(
    standardize            = 'lower_and_strip_punctuation',
    max_tokens             = VOCAB_SIZE,
    output_mode            = 'int',
    output_sequence_length = SEQUENCE_LENGTH
)

def model_builder(hp):
    """Build machine learning model"""
    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)

    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, hp['embedding_dim'], name='embedding')(x)
    x = layers.Bidirectional(layers.LSTM(hp['lstm_units']))(x)
    for _ in range(hp['num_layers']):
        x = layers.Dense(hp['dense_units'], activation='relu')(x)
    x = layers.Dropout(hp['dropout_rate'])(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    model.compile(
        loss      = tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer = tf.keras.optimizers.Adam(hp['learning_rate']),
        metrics   = [tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features      = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        # get predictions using the transformed features
        return model(transformed_features)

    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs) -> None:
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    hp      = fn_args.hyperparameters['values']

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir, update_freq='batch'
    )

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor  = 'val_binary_accuracy',
        mode     = 'max',
        verbose  = 1,
        patience = 10
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor        = 'val_binary_accuracy',
        mode           = 'max',
        verbose        = 1,
        save_best_only = True
    )

    callbacks = [
        tensorboard_callback,
        early_stop_callback,
        model_checkpoint_callback
    ]

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, hp['tuner/epochs'])
    val_set   = input_fn(fn_args.eval_files,  tf_transform_output, hp['tuner/epochs'])

    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)
        ]]
    )

    # Build the model
    model = model_builder(hp)

    # Train the model
    model.fit(
        x                = train_set,
        validation_data  = val_set,
        callbacks        = callbacks,
        steps_per_epoch  = fn_args.train_steps,
        validation_steps = fn_args.eval_steps,
        epochs           = hp['tuner/epochs']
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape = [None],
                dtype = tf.string,
                name  = 'examples'
            )
        )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format = 'tf',
        signatures  = signatures
    )


# %%
trainer = Trainer(
    module_file     = os.path.abspath(TRAINER_MODULE_FILE),
    examples        = transform.outputs['transformed_examples'],
    transform_graph = transform.outputs['transform_graph'],
    schema          = schema_gen.outputs['schema'],
    hyperparameters = tuner.outputs['best_hyperparameters'],
    train_args      = trainer_pb2.TrainArgs(splits=['train']),
    eval_args       = trainer_pb2.EvalArgs(splits=['eval'])
)

context.run(trainer)

# %% [markdown]
# ## Resolver

# %%
model_resolver = Resolver(
    strategy_class = LatestBlessedModelStrategy,
    model          = Channel(type=Model),
    model_blessing = Channel(type=ModelBlessing)
).with_id('Latest_blessed_model_resolver')

context.run(model_resolver)

# %% [markdown]
# ## Evaluator

# %%
eval_config = tfma.EvalConfig(
    model_specs   = [tfma.ModelSpec(label_key = 'Category')],
    slicing_specs = [tfma.SlicingSpec()],
    metrics_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name = 'ExampleCount'),
            tfma.MetricConfig(class_name = 'AUC'),
            tfma.MetricConfig(class_name = 'FalsePositives'),
            tfma.MetricConfig(class_name = 'TruePositives'),
            tfma.MetricConfig(class_name = 'FalseNegatives'),
            tfma.MetricConfig(class_name = 'TrueNegatives'),
            tfma.MetricConfig(class_name = 'BinaryAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold = tfma.GenericValueThreshold(
                        lower_bound = {'value': 0.5}
                    ),
                    change_threshold = tfma.GenericChangeThreshold(
                        direction = tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute  = {'value': 0.0001}
                    )
                )
            )
        ])
    ]
)

# %%
evaluator = Evaluator(
    examples       = example_gen.outputs['examples'],
    model          = trainer.outputs['model'],
    baseline_model = model_resolver.outputs['model'],
    eval_config    = eval_config
)

context.run(evaluator)

# %% [markdown]
# ### Visualisasi Evaluator

# %%
eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(
    tfma_result
)

# %% [markdown]
# ## Pusher
# deploy ke tf serving

# %%
pusher = Pusher(
model=trainer.outputs['model'],
model_blessing=evaluator.outputs['blessing'],
push_destination=pusher_pb2.PushDestination(
    filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory='serving_model_dir/spam-email-model'))
 
)
 
context.run(pusher)

# %%
!pip freeze > requirements.txt

# %%



