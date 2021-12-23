import tqdm
from tensorflow import (clip_by_value, constant,
                        expand_dims, tensor_scatter_nd_update,
                        transpose, zeros)
from tensorflow.dtypes import cast, float64, int64
from tensorflow.math import abs, exp, pow, reduce_sum, square
from tensorflow_probability import distributions


# pylint: disable = C0103
@staticmethod
def _moments_acc(n_teachers, votes, lap_scale, l_list):
    q = (2 + lap_scale * abs(2 * votes - n_teachers))/(4 * exp(lap_scale * abs(2 * votes - n_teachers)))

    update = []
    for l in l_list:
        clip = 2 * square(lap_scale) * l * (l + 1)
        t = (1 - q) * pow((1 - q) / (1 - exp(2 * lap_scale) * q), l) + q * exp(2 * lap_scale * l)
        update.append(reduce_sum(clip_by_value(t, clip_value_min=-clip, clip_value_max=clip)))
    return cast(update, dtype=float64)

def _pate_voting(self, data, netTD, lap_scale):
    ## Faz os votos dos teachers (1/0) netTD para cada record em data e guarda em results
    results = zeros([len(netTD), self.batch_size], dtype=int64)
    # print(results)
    for i in range(len(netTD)):
        output = netTD[i](data, training=True)
        pred = transpose(cast((output > 0.5), int64))
        # print(pred)
        results = tensor_scatter_nd_update(results, constant([[i]]), pred)
        # print(results)

    #guarda o somatorio das probabilidades atribuidas por cada disc a cada record (valores entre 0 e len(netTD))
    clean_votes = expand_dims(cast(reduce_sum(results, 0), dtype=float64), 1)
    # print("clean_votes",clean_votes)
    noise_sample = distributions.Laplace(loc=0, scale=1/lap_scale).sample(clean_votes.shape)
    # print("noise_sample", noise_sample)
    noisy_results = clean_votes + cast(noise_sample, float64)
    noisy_labels = cast((noisy_results > len(netTD)/2), float64)

    return noisy_labels, clean_votes
