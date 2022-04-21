from pmlb import fetch_data

from ydata_synthetic.synthesizers.regular import PATEGAN
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

model = PATEGAN

#Load data and define the data processor parameters
data = fetch_data('adult')
num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass','education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'native-country', 'target']

print(data.head())

#Defining the training parameters

noise_dim = 128
dim = 128
batch_size = 50

log_step = 100
epochs = 300+1
learning_rate = [5e-4, 3e-3]
beta_1 = 0.5
beta_2 = 0.9
models_dir = './cache'

gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           layers_dim=dim)

#  PATEGAN specific arguments
n_moments = 100
n_teacher_iters = 5
n_student_iters = 5
n_teachers = min(int(len(data)/1e3), 100)
##  Privacy/utility tradeoff specification
target_delta = 1e-3
target_epsilon = 1e-1
lap_scale = 1e-4

synthesizer = model(gan_args, n_teachers, target_delta, target_epsilon)
synthesizer.train(data, num_cols, cat_cols,
                  n_teacher_iters, n_student_iters, n_moments, lap_scale)

synthesizer.save('pate_test.pkl')

synthesizer = model.load('pate_test.pkl')
synth_data = synthesizer.sample(1000)
