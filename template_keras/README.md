# Template for deep learning experiments with keras

This python package aim to provide with a simple solution to train any network quickly and be able to re-run the experiment with the same parameters if required.

This template defines the minimum parameters required to run the 4 main files provided. All the required parameters are defined through inheritance and all the class defined by the user should inherit the corresponding file if available in the template.

The following environnement variable are required by the different scripts:

- EXPERIMENTS_OUTPUT_DIRECTORY: were the experiments will be saved
- LOG_DIRECTORY: were all the log will be written
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
