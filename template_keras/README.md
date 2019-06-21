# Template for deep learning experiments with keras

This python package aim to provide with a simple solution to train any network quickly and be able to re-run the experiment with the same parameters if required.

This template defines the minimum parameters required to run the 4 main files provided. All the required parameters are defined through inheritance and all the class defined by the user should inherit the corresponding classes if available in the template.

The main template classes are:

- The TemplateConfiguration, used to define the config file that will hold all the parameters of a given experiment
- The TemplateDisplayer, used to define the class to display the data
- The TemplateEvaluator, used to define the class to evaluate an experiment
- The TemplateGenerator, used to define the class to generate the data

## mnist_example

The following environnement variable are required by the different scripts in mnist_example:

- EXPERIMENTS_OUTPUT_DIRECTORY: were the experiments will be saved
- LOG_DIRECTORY: were all the log will be written (this is to be used if a job id is specified when training)
- COMET_API_KEY: If using comet.ml, the API key to log the experiments

To run the training.py file, the following classes are required:

- a config file
- a generator

To run the evaluate.py file, the following classes are required:

- a config file
- a generator
- an evaluator

To run the inference.py file, the following classes are required:

- a config file
- a generator
- a displayer

To run the inference_time.py file, the following classes are required:

- a config file
- a generator
- an evaluator

### Run the example

For the example below, we will assume that a python env was created and activated before running the commands. We also assume that 

```bash
# Let us install the package
cd template_keras
python setup.py install

# We will run the mnist example, but first let's create the environment variables and some dummy directory to save the results to
mkdir -p /tmp/mnist_example/outputs
mkdir -p /tmp/mnist_example/logs

export EXPERIMENTS_OUTPUT_DIRECTORY=/tmp/mnist_example/outputs
export LOG_DIRECTORY=/tmp/mnist_example/logs

# We check everything worked well
echo $EXPERIMENTS_OUTPUT_DIRECTORY
echo $LOG_DIRECTORY

# Now we go to the mnist directory and install the required packages (you may want to change some stuff for tensorflow depending on your setup)
cd ../mnist_example/
pip install -r requirements.txt

# Let's run the command to train the network
# Use python training.py -h to get the description of the parameters
python training.py -c config/custom/

# Now that we have trained the network, we want to check the results
python evaluate.py /tmp/mnist_example/outputs/<experiment>/ /tmp/mnist_example/outputs/<experiment>/checkpoints/<weights_to_load>

# Maybe display some results
python inference.py /tmp/mnist_example/outputs/<experiment>/ /tmp/mnist_example/outputs/<experiment>/checkpoints/<weights_to_load>

# Maybe test the speed of the network
python inference_time.py experiment /tmp/mnist_example/outputs/<experiment>/ /tmp/mnist_example/outputs/<experiment>/checkpoints/<weights_to_load>
```