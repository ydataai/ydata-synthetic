"""
Discriminator models classes defintion
"""

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import ReLU, Dense, Flatten, Lambda, Concatenate
from tensorflow.keras.backend import squeeze

class Discriminator():
    """
    Main TS-Synthesizer Discriminator class.
    """
    def __init__(self,dis_dim=(200, 200, 200, 200), scope_name='discriminator', *args, **kwargs):
        super(Discriminator, self).__init__(name=scope_name, *args, **kwargs)
        self.dis_dim = dis_dim

    def build_model(self, feature_shape, attr_shape, batch_size):
        """
        Parameters
        ----------
        feature_shape Tuple with features input shape
        attr_shape Tuple with attributes input shape
        Returns
        ------
        """
        feature_input = Input(shape=feature_shape, batch_size=batch_size)
        attr_input = Input(shape=attr_shape, batch_size=batch_size)
        features = Flatten()(feature_input)
        attr = Flatten()(attr_input)
        concat = Concatenate(axis=1)([features, attr])

        for item in list(self.dis_dim):
            concat = Dense(item)(concat)
            concat = ReLU()(concat)
            dim = item

        output = Dense(1, input_shape=(dim,))(concat)
        output = Lambda(lambda x: squeeze(x, 1))(output)
        return Model(inputs=[feature_input, attr_input], outputs=output)

class AttrDiscriminator():
    """
    Attributes specific discriminator.
    Auxiliar Discriminator to improve TSGenerator results for attribute variables
    """
    def __init__(self, attr_dis_dim=(200, 200, 200, 200),
                 scope_name='attrDiscriminator', *args, **kwargs):
        super(AttrDiscriminator, self).__init__(name=scope_name, *args, **kwargs)
        self.attr_dis_dim = attr_dis_dim

    def build_model(self, input_shape, batch_size):
        """
        Parameters
        ----------
        input_shape Attributes input shape

        Returns A Keras Model that will be further during the training process
        -------
        """
        attr_input = Input(shape=input_shape, batch_size=batch_size)
        flattened = Flatten()(attr_input)

        for item in list(self.attr_dis_dim):
            flattened = Dense(item)(flattened)
            flattened = ReLU()(flattened)
            dim = item

        output = Dense(1, input_shape=(dim,))(flattened)
        output = Lambda(lambda x: squeeze(x, 1))(output)

        return Model(inputs=attr_input, outputs=output)
