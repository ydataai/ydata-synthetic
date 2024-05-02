from ydata_synthetic.synthesizers.base import ModelParameters, TrainParameters

"""
ModelParameters:
Defines the parameters required to initialize a synthesizer model.
"""
class ModelParameters(ModelParameters):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

