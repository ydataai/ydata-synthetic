import tensorflow as tf
import numpy as np


class RealDataSampler:
    """
    Class used to sample from real data.

    Args:
        data: Input data.
        metadata: Dataset columns metadata.
    """
    def __init__(self, data, metadata):
        super(RealDataSampler, self).__init__()
        self._data = data
        self._active_bits = []
        self._n_rows = len(data)

        for col_md in metadata:
            if col_md.discrete:
                col_active_bits = []
                for c in range(col_md.output_dim):
                    col_active_bits.append(np.nonzero(data[:, col_md.start_idx + c])[0])
                self._active_bits.append(col_active_bits)

    def sample(self, num_samples):
        """
        Samples from the entire dataset.

        Args:
            num_samples: Number of samples to be returned.
        """
        return self._data[np.random.choice(np.arange(self._n_rows), num_samples)]

    def sample_col(self, col_idx, opt_idx):
        """
        Samples a specific discrete column.

        Args:
            col_idx: Index of the column to be sampled.
            opt_idx: Index of the category.
        """
        idx = []
        for col, opt in zip(col_idx, opt_idx):
            idx.append(np.random.choice(self._active_bits[col][opt]))
        return self._data[idx]


class ConditionalSampler:
    """
    Class used to sample conditional vectors.

    Args:
        data: Input data.
        metadata: Dataset columns metadata.
        log_frequency: Whether to apply log frequency or not.
    """
    def __init__(self, data=None, metadata=None, log_frequency=None):
        if data is None:
            return
        self._active_bits = []
        max_interval = 0
        counter = 0

        for col_md in metadata:
            if col_md.discrete:
                max_interval = max(max_interval, col_md.end_idx - col_md.start_idx)
                self._active_bits.append(np.argmax(data[:, col_md.start_idx:col_md.end_idx], axis=-1))
                counter += 1

        self._interval = []
        self._n_col = 0
        self._n_opt = 0
        self._probabilities = np.zeros((counter, max_interval))

        for col_md in metadata:
            if col_md.discrete:
                col_active_bits_sum = np.sum(data[:, col_md.start_idx:col_md.end_idx], axis=0)
                if log_frequency:
                    col_active_bits_sum = np.log(col_active_bits_sum + 1)
                col_active_bits_sum = col_active_bits_sum / np.sum(col_active_bits_sum)
                self._probabilities[self._n_col, :col_md.output_dim] = col_active_bits_sum
                self._interval.append((self._n_opt, col_md.output_dim))
                self._n_opt += col_md.output_dim
                self._n_col += 1

        self._interval = np.asarray(self._interval)        

    @property
    def output_dimensions(self):
        """
        Returns the dimensionality of the conditional vectors.
        """
        return self._n_opt

    def sample(self, batch_size, from_active_bits=False):
        """
        Samples conditional vectors.

        Args:
            batch_size: Batch size.
            from_active_bits: Whether to directly sample from active bits or not.
        """
        if self._n_col == 0:
            return None
        
        col_idx = np.random.choice(np.arange(self._n_col), batch_size)
        cond_vector = np.zeros((batch_size, self._n_opt), dtype='float32')

        if from_active_bits:
            for i in range(batch_size):
                pick = int(np.random.choice(self._active_bits[col_idx[i]]))
                cond_vector[i, pick + self._interval[col_idx[i], 0]] = 1
            return cond_vector

        mask = np.zeros((batch_size, self._n_col), dtype='float32')
        mask[np.arange(batch_size), col_idx] = 1
        prob = self._probabilities[col_idx]
        rand = np.expand_dims(np.random.rand(prob.shape[0]), axis=1)
        opt_idx = (prob.cumsum(axis=1) > rand).argmax(axis=1)
        opt = self._interval[col_idx, 0] + opt_idx
        cond_vector[np.arange(batch_size), opt] = 1
        return cond_vector, mask, col_idx, opt_idx
    
class ConditionalLoss:
    """
    Conditional loss utils.
    """
    @staticmethod
    def compute(data, cond_vector, mask, metadata):
        """
        Computes the conditional loss.

        Args:
            data: Input data.
            cond_vector: Conditional vector.
            mask: Mask vector.
            metadata: Dataset columns metadata.
        """
        shape = tf.shape(mask)
        cond_loss = tf.zeros(shape)
        start_cat = 0
        counter = 0
        for col_md in metadata:
            if col_md.discrete:
                end_cat = start_cat + col_md.output_dim
                data_log_softmax = data[:, col_md.start_idx:col_md.end_idx]
                cond_vector_am = tf.math.argmax(cond_vector[:, start_cat:end_cat], 1)
                loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    cond_vector_am, data_log_softmax), [-1, 1])
                cond_loss = tf.concat(
                    [cond_loss[:, :counter], loss, cond_loss[:, counter+1:]], 1)
                start_cat = end_cat
                counter += 1

        return tf.reduce_sum(cond_loss * mask) / tf.cast(shape[0], dtype=tf.float32)
