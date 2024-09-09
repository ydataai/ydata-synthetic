# Great Expectations

[Great Expectations](https://greatexpectations.io) is a Python-based open-source library for validating, documenting, and profiling your data. It helps you to maintain data quality and improve communication about data between teams. With Great Expectations, you can assert what you expect from the data you load and transform, and catch data issues quickly – Expectations are basically *unit tests for your data.*

## About Great Expectations
*Expectations* are assertions about your data. In Great Expectations, those assertions are expressed in a declarative language in the form of simple, human-readable Python methods. For example, in order to assert that you want values in a column `passenger_count` in your dataset to be integers between 1 and 6, you can say:

```python
expect_column_values_to_be_between(column="passenger_count", min_value=1, max_value=6)
```

Great Expectations then uses this statement to validate whether the column `passenger_count` in a given table is indeed between 1 and 6, and returns a success or failure result. The library currently provides [several dozen highly expressive built-in Expectations](https://greatexpectations.io/expectations/), and allows you to write [custom Expectations](https://docs.greatexpectations.io/docs/guides/expectations/custom_expectations_lp/).

Great Expectations renders Expectations to clean, human-readable documentation called *Data Docs*. These HTML docs contain both your Expectation Suites as well as your data validation results each time validation is run – think of it as a continuously updated data quality report.

## Validating your Synthetic Data with Great Expectations

!!! note `Outdated`
    From ydata-synthetic vx onwards this example will no longer work. Please check `ydata-sdk` and [synthetic data generation examples](https://docs.fabric.ydata.ai/latest/sdk/examples/synthesize_tabular_data/).

#### 1. Install the required libraries:
We recommend you create a virtual environment and install ydata-synthetic and great-expectations by running the following command on your terminal.

```bash
pip install ydata-synthetic great-expectations
```

#### 2. Generate your Synthetic Data:
In this example, we'll use CTGAN to synthesize samples from the [Adult Census Income](https://www.kaggle.com/datasets/uciml/adult-census-income?resource=download) dataset:

```python
from pmlb import fetch_data

from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Load data and define the data processor parameters
data = fetch_data('adult')
num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass','education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'native-country', 'target']

# Defining the training parameters
batch_size = 500
epochs = 500+1
learning_rate = 2e-4
beta_1 = 0.5
beta_2 = 0.9

ctgan_args = ModelParameters(batch_size=batch_size,
                             lr=learning_rate,
                             betas=(beta_1, beta_2))

train_args = TrainParameters(epochs=epochs)
synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

# Sample for the trained synthesizer and save the synthetic data
synth_data = synth.sample(1000)
synth_data.to_csv('data/adult_synthetic.csv', index=False)
```

#### 3. Create a Data Context and Connect to Data:
Import the `great_expectations` module, create a data context, and connect to your synthetic data:

```python
import great_expectations as gx

# Initialize data context
context = gx.get_context()

# Connect to the synthetic data
validator = context.sources.pandas_default.read_csv(
    "data/adult_synthetic.csv"
)
```

#### 4. Create Expectations:
You can create Expectation Suites by writing out individual statements, such as the ones below, by using [Profilers and Data Assistants](https://docs.greatexpectations.io/docs/guides/expectations/profilers_data_assistants_lp) or even [Custom Profilers](https://docs.greatexpectations.io/docs/guides/expectations/advanced/how_to_create_a_new_expectation_suite_using_rule_based_profilers/).

```python
# Create expectations
validator.expect_column_values_to_not_be_null("age")
validator.expect_column_values_to_be_between("workclass", auto=True)
validator.save_expectation_suite()
```

#### 5. Validate Data
To validate your data, define a checkpoint and examine the data to determine if it matches the defined Expectations:

```python
# Validate the synthetic data
checkpoint = context.add_or_update_checkpoint(
    name="synthetic_data_checkpoint",
    validator=validator,
)
```
You can run the validations results:

```python
checkpoint_result = checkpoint.run()
```

And use the following code to view an HTML representation of the Validation results:

```python
context.view_validation_result(checkpoint_result)
```

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=dd69a9f9-0901-4cb4-9e56-b1e69877dca1" />
