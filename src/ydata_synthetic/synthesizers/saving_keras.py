import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

from tensorflow.keras import Model

def unpack(model, training_config, weights):
    restored_model = tf_keras.layers.deserialize(model)
    if training_config is not None:
        restored_model.compile(**tf_keras.saving.saving_utils.compile_args_from_training_config(training_config))
    restored_model.set_weights(weights)
    return restored_model

def make_keras_picklable():
    def __reduce__(self):
        model_metadata = tf_keras.saving.saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = tf_keras.layers.serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__=__reduce__
