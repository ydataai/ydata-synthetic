from tensorflow.keras import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

def unpack(model, training_config, weights):
    # Deserialize the model from its JSON string representation
    restored_model = deserialize(model)
    
    # Compile the model with the training configuration, if provided
    if training_config is not None:
        restored_model.compile(**saving_utils.compile_args_from_training_config(training_config))
    
    # Set the weights of the model to the provided values
    restored_model.set_weights(weights)
    
    # Return the restored and configured model
    return restored_model

def make_keras_picklable():
    # Save the original __reduce__ method of the Model class
    original_reduce = Model.__reduce__
    
    # Define a new __reduce__ method for the Model class
    def __reduce__(self):
        # Save the model as a JSON string
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        
        # Save the model's weights
        weights = self.get_weights()
        
        # Return a tuple that can be used to recreate the model
        return (unpack, (model, training_config, weights))
    
    # Replace the __reduce__ method of the Model class with the new one
    cls = Model
    cls.__reduce__ = __reduce__
