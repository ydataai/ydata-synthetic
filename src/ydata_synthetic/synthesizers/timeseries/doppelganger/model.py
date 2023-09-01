from pandas import DataFrame
import tensorflow as tf
import os
from joblib import dump, load

from ydata_synthetic.synthesizers.timeseries.doppelganger.network import DoppelGANgerGenerator, AttrDiscriminator, Discriminator
from ydata_synthetic.synthesizers.timeseries.doppelganger.doppelganger import DoppelGANgerNetwork
from ydata_synthetic.synthesizers.base import BaseGANModel, ModelParameters, TrainParameters
from ydata_synthetic.preprocessing.timeseries.doppelganger_processor import DoppelGANgerProcessor

class DoppelGANger(BaseGANModel):
    """
    DoppelGANger model.
    Based on the paper https://dl.acm.org/doi/pdf/10.1145/3419394.3423643.

    Args:
        model_parameters: Parameters used to create the DoppelGANger model.
    """
    __MODEL__ = 'DoppelGANger'

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters)
        self._model_parameters = model_parameters
        self._gan_model = None
        self._tf_session = None
        self._sequence_length = None
        tf.compat.v1.disable_eager_execution()

    def fit(self, data: DataFrame,
            train_arguments: TrainParameters,
            num_cols: list[str] | None = None,
            cat_cols: list[str] | None = None):
        """
        Fits the DoppelGANger model.

        Args:
            data: A pandas DataFrame with the data to be synthesized.
            train_arguments: DoppelGANger training arguments.
            num_cols: List of columns to be handled as numerical
            cat_cols: List of columns to be handled as categorical
        """
        super().fit(data=data, num_cols=num_cols, cat_cols=cat_cols, train_arguments=train_arguments)

        self._sequence_length = train_arguments.sequence_length
        self._sample_length = train_arguments.sample_length
        self._rounds = train_arguments.rounds

        if data.shape[0] % self._sequence_length != 0:
            raise ValueError("The number of samples must be a multiple of the sequence length.")

        if self._sequence_length % self._sample_length != 0:
            raise ValueError("The sequence length must be a multiple of the sample length.")

        data_features, data_attributes = self.processor.transform(data)
        measurement_cols_metadata = self.processor.measurement_cols_metadata
        attribute_cols_metadata = self.processor.attribute_cols_metadata

        generator = DoppelGANgerGenerator(
            feed_back=False,
            noise=True,
            use_tanh=self.use_tanh,
            measurement_cols_metadata=measurement_cols_metadata,
            attribute_cols_metadata=attribute_cols_metadata,
            sample_len=self._sample_length)
        discriminator = Discriminator()
        attr_discriminator = AttrDiscriminator()

        self._tf_session = tf.compat.v1.Session()
        with self._tf_session.as_default() as sess:
            self._gan_model = DoppelGANgerNetwork(
                sess=sess,
                epoch=train_arguments.epochs,
                batch_size=self.batch_size,
                data_feature=data_features,
                data_attribute=data_attributes,
                attribute_cols_metadata=attribute_cols_metadata,
                sample_len=self._sample_length,
                generator=generator,
                discriminator=discriminator,
                rounds=self._rounds,
                attr_discriminator=attr_discriminator,
                d_gp_coe=self.gp_lambda,
                attr_d_gp_coe=self.gp_lambda,
                g_attr_d_coe=self.gp_lambda,
                num_packing=self.pac,
                attribute_latent_dim=self.latent_dim,
                feature_latent_dim=self.latent_dim,
                fix_feature_network=False,
                g_lr=self.g_lr,
                g_beta1=self.beta_1,
                d_lr=self.d_lr,
                d_beta1=self.beta_1,
                attr_d_lr=self.d_lr,
                attr_d_beta1=self.beta_1)
            self._gan_model.build()
            self._gan_model.train()

    def sample(self, n_samples: int):
        """
        Samples new data from the DoppelGANger.

        Args:
            n_samples: Number of samples to be generated.
        """
        if n_samples <= 0:
            raise ValueError("Invalid number of samples.")

        real_attribute_input_noise = self._gan_model.gen_attribute_input_noise(n_samples)
        addi_attribute_input_noise = self._gan_model.gen_attribute_input_noise(n_samples)
        length = int(self._sequence_length / self._sample_length)
        feature_input_noise = self._gan_model.gen_feature_input_noise(n_samples, length=length)
        input_data = self._gan_model.gen_feature_input_data_free(n_samples)

        with self._tf_session.as_default() as sess:
            self._gan_model.sess = sess
            data_features, data_attributes, gen_flags, _ = self._gan_model.sample_from(
                real_attribute_input_noise, addi_attribute_input_noise,
                feature_input_noise, input_data)

        return self.processor.inverse_transform(data_features, data_attributes, gen_flags)

    def save(self, path):
        """
        Save the DoppelGANger model in a directory.

        Args:
            path: Path of the directory where the files will be saved.
        """
        saver = tf.compat.v1.train.Saver()
        with self._tf_session.as_default() as sess:
            saver.save(sess, os.path.join(path, "doppelganger"), write_meta_graph=False)
        self._gan_model.save(os.path.join(path, "doppelganger_network.pkl"))
        dump({
            "processor": self.processor.__dict__,
            "measurement_cols_metadata": self.processor.measurement_cols_metadata,
            "attribute_cols_metadata": self.processor.attribute_cols_metadata,
            "_sequence_length": self._sequence_length,
            "_sample_length": self._sample_length
        }, os.path.join(path, "doppelganger_metadata.pkl"))

    @staticmethod
    def load(path):
        """
        Load the DoppelGANger model from a directory.
        Only the required components to sample new data are loaded.

        Args:
            class_dict: Path of the directory where the files were saved.
        """
        dp_model = DoppelGANger(ModelParameters())
        dp_network_parms = load(os.path.join(path, "doppelganger_network.pkl"))
        dp_metadata = load(os.path.join(path, "doppelganger_metadata.pkl"))

        dp_model.processor = DoppelGANgerProcessor()
        dp_model.processor.__dict__ = dp_metadata["processor"]
        dp_model._sequence_length = dp_metadata["_sequence_length"]
        dp_model._sample_length = dp_metadata["_sample_length"]

        generator = DoppelGANgerGenerator(
            feed_back=False,
            noise=True,
            measurement_cols_metadata=dp_metadata["measurement_cols_metadata"],
            attribute_cols_metadata=dp_metadata["attribute_cols_metadata"],
            sample_len=dp_network_parms["sample_len"])
        discriminator = Discriminator()
        attr_discriminator = AttrDiscriminator()

        with tf.compat.v1.Session().as_default() as sess:
            dp_model._gan_model = DoppelGANgerNetwork(
                sess=sess,
                epoch=dp_network_parms["epoch"],
                batch_size=dp_network_parms["batch_size"],
                data_feature=None,
                data_attribute=None,
                attribute_cols_metadata=dp_metadata["attribute_cols_metadata"],
                sample_len=dp_network_parms["sample_len"],
                generator=generator,
                discriminator=discriminator,
                rounds=dp_network_parms["rounds"],
                attr_discriminator=attr_discriminator,
                d_gp_coe=dp_network_parms["d_gp_coe"],
                attr_d_gp_coe=dp_network_parms["attr_d_gp_coe"],
                g_attr_d_coe=dp_network_parms["g_attr_d_coe"],
                num_packing=dp_network_parms["num_packing"],
                attribute_latent_dim=dp_network_parms["attribute_latent_dim"],
                feature_latent_dim=dp_network_parms["feature_latent_dim"],
                fix_feature_network=dp_network_parms["fix_feature_network"],
                g_lr=dp_network_parms["g_lr"],
                g_beta1=dp_network_parms["g_beta1"],
                d_lr=dp_network_parms["d_lr"],
                d_beta1=dp_network_parms["d_beta1"],
                attr_d_lr=dp_network_parms["attr_d_lr"],
                attr_d_beta1=dp_network_parms["attr_d_beta1"])

            dp_model._gan_model.sample_time = dp_network_parms["sample_time"]
            dp_model._gan_model.sample_feature_dim = dp_network_parms["sample_feature_dim"]
            dp_model._gan_model.sample_attribute_dim = dp_network_parms["sample_attribute_dim"]
            dp_model._gan_model.sample_real_attribute_dim = dp_network_parms["sample_real_attribute_dim"]
            dp_model._gan_model.build()

            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, tf.compat.v1.train.latest_checkpoint(path))
            dp_model._tf_session = sess

        return dp_model
