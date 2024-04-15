# meta-template


# Requirements

1. Create a python venv to run code in (the code was developed with Python 3.9.12, but the specific version probably doesn't matter much, as long as it is some version of Python 3):
```
python -m venv .venv
```

2. Activate the venv (the venv should be active any time you are running code from the repo).
```
source .venv/bin/activate
```

3. Install requirements
```
pip install -U pip setuptools wheel
pip install torch
pip install higher
pip install numpy
pip install transformers
pip install jsonlines
```


# Quickstart
1. Here is a command you can use to meta-train a logical rule learneri (you may need to first create directories for logs and weights by running `mkdir logs` and `mkdir weights`):
```
python meta_train.py --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset dnf --min_n_features 4 --max_n_features 4 --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name tmp --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1
```

2. To evaluate it, you can run the same command but with 2 changes: (i) add the flags `--eval --eval_table3` to the command; `--eval` says to only evaluate, not train, and `--eval_table3` says to perform the evaluation corresponding to Table 3 in the Rational Rules paper. (ii) When the model was trained with the previous command, an index would have been added to the model name; to evaluate it, you need to add this index to the model name you list. Assuming this was the first time you ran the above command, the index would be zero. Thus, the full evaluation command would be:
```
python meta_train.py --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset dnf --min_n_features 4 --max_n_features 4 --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name tmp_0 --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1 --eval --eval_table3
```
Ideally, the outputs from the model (which are shown in the last column) should display similar trends as the humans and the RR_DNF model (which are in the third-to-last and second-to-last columns).

3. You can also see how a randomly-initialized neural network performs. It will likely do much worse than the meta-trained neural network. To do this, run the same evaluation command but use a model name that starts with `random`, e.g.:
```
python meta_train.py --n_meta_train 10000 --n_meta_valid 100 --n_meta_test 100 --dataset dnf --min_n_features 4 --max_n_features 4 --min_n_train 9 --max_n_train 9 --train_batch_size 1 --b 3 --n_hidden 128 --n_layer 5 --dropout 0.1 --n_epochs 1 --eval_every 100 --learning_rate 0.0005 --inner_lr 0.1 --model_name random_tmp_0 --weight_dir weights/ --log_dir logs/ --epochs_per_episode 1 --eval --eval_table3

```


4. Parameters to play around with:
- If we decide to make the model more general, adjust `--min_n_features`, `--max_n_features`, `--min_n_train`, and `--max_n_train`.
- `--train_batch_size`, `--inner_lr`; probably the most important
- `--b`: corresponds to b hyperparameter from the paper
- Architectural hyperparameters: `--n_hidden`, `--n_layer`, `--dropout`
- `--epochs_per_episode`: Probably not worth adjusting


# Description of the pipeline

1. First, you need to create a dataloader. The dataloader should contain 3 data splits (training, validation, and test). Each split should be an iterator where, as you iterate over it, it returns one batch at a time. Dataloaders are defined in `dataloading.py`

    a. For standard training, each batch will be one input to your model. It should be a dictionary containing the input as well as the target output that this input should have. 

    b. For meta training, each batch corresponds to one episode of metatraining. Therefore, it should contain both the training set for this episode and the test set for this episode. Specifically, the batch will be a dictionary with the following keys: `training_batches`: a list of batches (set up just like a standard batch in 1a - each of these batches should contain the model's input for that batch and the target output). `test_input_ids`: The inputs for the test set for this episode. `test_labels`: The labels for the test set for this episode. An example of a meta dataloader is `MetaLogicDataset`.

    c. It may be necessary to also define a dataset iterator (like `dnf_dataset`) that samples a dataset from some prior; potential drawing upon another file that implements sampling from this prior, like `dnf_grammar`.

2. Then, you need to create a model. The model should take in a single standard batch (as defined in 1a: a dictionary containing the inputs and the target outputs for that batch). Then it should return the model's predicted output and the model's loss (when its predicted output is compared to the target output). `MLPClassifier` gives an example of a model. 

3. Then, you need to train the model. This is done with a trainer (or metatrainer) from `training.py`, which takes in the model and dataset (and training parameters like the learning rate or number of epochs) and trains the model on that dataset. If your model and dataset are set up as described above, then you might not need to change or add anything to `training.py`.

4. Finally, you need to evaluate your model. For this, you will have to write functions that define whatever evaluations you want to run.

5. To put it all together, you can create a single script (like `meta_train.py`) which first instantiates the dataset and the model, then trains the model, and then evaluates the model.

# Description of files

- `meta_train.py`: Meta-train a model (or evaluate a trained model)
- `dataloading.py`: Code for loading and preparing data
- `dataset_iterators.py`: Functions that yield a rule sampled from the prior
- `evaluations.py`: Evaluation functions (beyond the loss automatically returned by models)
- `lr_scheduler.py`: Functions for learning rate scheduling
- `models.py`: Defining model architectures
- `utils.py`: Some miscellaneous helper functions
- `dnf_grammar.py`: Specifies the RR_DNF prior from the Rational Rules paper
- `training.py`: Classes for (meta-)training models
