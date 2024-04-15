from packaging import version
from enum import Enum
import logging
import math
import os
from collections import Counter
import copy
import random
import sys

import torch
import higher

from itertools import chain

from transformers import PreTrainedModel, GPT2LMHeadModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

from lr_scheduler import *

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# Based closely on Hugging Face's Trainer class
class Trainer:

    def __init__(self, model=None, train_datasplit=None, eval_datasplit=None,
            max_grad_norm=1.0, n_epochs=None, patience=None, lr_decay_patience=1, weight_decay=0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, learning_rate=5e-5, 
            lr_scheduler_type="linear", warmup_steps=0, eval_every=None, log_every=None, tokenizer=None, vary_train_batch_size=False):

        # Model attributes
        self.model = model

        # Data attributes
        self.train_datasplit = train_datasplit
        self.eval_datasplit = eval_datasplit
        self.train_size = len(train_datasplit)
        self.tokenizer = tokenizer
        self.vary_train_batch_size = vary_train_batch_size

        # Learning attributes
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.patience = patience
        self.lr_decay_patience = lr_decay_patience
        self.stop = False # Whether we've hit the patience

        # Optimizer attributes
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps

        # Logging attributes
        self.eval_every = eval_every
        self.log_every = log_every

        # Early stopping attributes
        self.best_loss = math.inf
        self.updates_since_improved = 0
        self.lr_decays_since_improved = 0

 
    def create_optimizer_and_scheduler(self, num_training_steps=None):
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.trainable_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.trainable_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_kwargs = {
                    "betas": (self.adam_beta1, self.adam_beta2),
                    "eps": self.adam_epsilon,
                    "lr": self.learning_rate,
                }
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
        
        self.lr_scheduler = get_scheduler(
                self.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=num_training_steps,
            )

    
    def training_step(self, model, inputs):
       
        # Put the model in training mode
        model.train()

        # Compute the loss
        if self.vary_train_batch_size:
            loss = model(inputs, reduction="sum")["loss"]
        else:
            loss = model(inputs, reduction="mean")["loss"]

        loss.backward()

        return loss.detach()

    def prediction_step(self, model, inputs):

        # Put the model in evaluation mode
        model.eval()

        # Compute the loss
        with torch.no_grad():
            outputs = model(inputs)
            loss = outputs["loss"].mean().detach()
            acc = outputs["accuracy"]

        return loss, acc


    def evaluate(self, eval_datasplit=None, name="Validation", save=False):

        if eval_datasplit is None:
            eval_datasplit = self.eval_datasplit

        eval_datasplit.reset()

        total_eval_loss = 0
        total_tokens = 0
        total_eval_acc = 0
        for batch_index, batch in enumerate(eval_datasplit):
            loss, acc = self.prediction_step(self.model, batch)
            loss = loss.item()

            # Ignore padding tokens, which have a label of -100
            tokens_in_batch = torch.sum(batch["labels"].view(-1) >= 0).item()

            # Multiply by tokens in batch to control for varying length batches
            total_eval_loss += loss*tokens_in_batch
            total_eval_acc += acc*tokens_in_batch
            total_tokens += tokens_in_batch

        # Divide by total tokens to get average
        avg_eval_loss = total_eval_loss / total_tokens
        logging.info(name + " loss: " + str(avg_eval_loss))

        avg_eval_acc = total_eval_acc / total_tokens
        logging.info(name + " accuracy: " + str(avg_eval_acc))

        # Enable learning rate decrease
        if save and avg_eval_loss >= self.best_loss and self.patience is not None:
            self.updates_since_improved += 1

            if self.updates_since_improved >= self.patience:
                self.lr_decays_since_improved += 1
                if self.lr_decays_since_improved >= self.lr_decay_patience:
                    self.stop = True

                # Restarting with a smaller learning rate
                self.updates_since_improved = 0
                self.learning_rate = self.learning_rate * 0.5
                logging.info("REDUCING LEARNING RATE TO " + str(self.learning_rate))
                self.model.load()
                self.create_optimizer_and_scheduler(num_training_steps=self.max_steps)

        if save and avg_eval_loss < self.best_loss:
            self.updates_since_improved = 0
            self.best_loss = avg_eval_loss

            # Save checkpoint
            self.model.save()

    def train(self):

        max_steps = self.n_epochs * self.train_size
        self.max_steps = max_steps
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.model.train()

        tr_loss = 0
        self.model.zero_grad()

        total_updates = 0


        for epoch in range(self.n_epochs):

            # Reset training set
            self.train_datasplit.reset()

            if self.stop:
                # We've hit the patience, so we should stop training
                break

            for batch_index, batch in enumerate(self.train_datasplit):
                if total_updates % self.eval_every == 0:
                    self.evaluate(save=True)

                if self.stop:
                    # We've hit the patience, so we should stop training
                    break

                if total_updates % self.log_every == 0:
                    logging.info("Training step " + str(total_updates) + " out of " + str(max_steps) + "; Epoch " + str(epoch) + "; Learning rate: " + str(self.lr_scheduler.get_last_lr()[0]))

                tr_loss += self.training_step(self.model, batch)

                if self.max_grad_norm is not None and self.max_grad_norm > 0:

                    if hasattr(self.optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        self.optimizer.clip_grad_norm(self.max_grad_norm)
                    else:
                        # Revert to normal clipping otherwise
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.model.zero_grad()
                
                self.lr_scheduler.step()

                total_updates += 1



class MetaTrainer(Trainer):

    def __init__(self, inner_lr=1e-1, multi_step_loss=False, epochs_per_episode=1, **kwargs):
        super(MetaTrainer, self).__init__(**kwargs)

        self.inner_lr = inner_lr
        sys.setrecursionlimit(10000)
        self.inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        sys.setrecursionlimit(100)

        self.multi_step_loss = multi_step_loss
        
        self.epochs_per_episode = epochs_per_episode


    def training_step(self, model, inputs):

        with torch.backends.cudnn.flags(enabled=False):

            model.train()
            test = {"input_ids" : inputs["test_input_ids"], "labels" : inputs["test_labels"]}


            with higher.innerloop_ctx(model, self.inner_opt, copy_initial_weights=False) as (fmodel, diffopt):

                outer_loss = 0

                # Train on the training set for this episode
                for epoch in range(self.epochs_per_episode):
                    for mini_batch in inputs["train_batches"]:

                        inner_loss = fmodel(mini_batch)["loss"]

                        diffopt.step(inner_loss)

                        if self.multi_step_loss:
                            this_loss = 0
                            for test_batch in inputs["test_batches"]:
                                if self.vary_train_batch_size:
                                    this_loss += fmodel(test_batch, reduction="sum")["loss"]
                                else:
                                    this_loss += fmodel(test_batch, reduction="mean")["loss"]
                        
                            this_loss.backward(retain_graph=True)
                            outer_loss += this_loss.detach()
            
                if not self.multi_step_loss:
                    # Evaluate on the test set for this episode
                    outer_loss = fmodel(test)["loss"]
                    outer_loss.backward()
                    return outer_loss.detach()
            
                else:
                    return outer_loss



    def prediction_step(self, model, inputs):
        
        with torch.backends.cudnn.flags(enabled=False):
        
            model.eval()
            test = {"input_ids" : inputs["test_input_ids"], "labels" : inputs["test_labels"]}

            fmodel = copy.deepcopy(model)
            diffopt = torch.optim.SGD(fmodel.parameters(), lr=self.inner_lr)

            # Train on the training set for this episode
            for epoch in range(self.epochs_per_episode):
                for mini_batch in inputs["train_batches"]:
                    inner_loss = fmodel(mini_batch)["loss"]
                    inner_loss.backward()
                    diffopt.step()
                    fmodel.zero_grad()

            # Evaluate on the test set for this episode
            outputs = fmodel(test)
            outer_loss = outputs["loss"].detach()
            outer_acc = outputs["accuracy"]

            return outer_loss, outer_acc


# Uses the same data as a MetaTrainer, but only uses standard (pre-)training, not MAML
class PseudoMetaTrainer(MetaTrainer):

    def __init__(self, **kwargs):
        super(PseudoMetaTrainer, self).__init__(**kwargs)


    def training_step(self, model, inputs):

        model.train()
        test = {"input_ids" : inputs["test_input_ids"], "labels" : inputs["test_labels"]}

        
        outer_loss = 0

        # Train on the training set for this episode
        for mini_batch in inputs["train_batches"]:

            inner_loss = model(mini_batch)["loss"]
            inner_loss.backward()

            if self.multi_step_loss:
                this_loss = 0
                for test_batch in inputs["test_batches"]:
                    this_loss += model(test_batch)["loss"]
                        
                this_loss.backward(retain_graph=True)
                outer_loss += this_loss.detach()
            
        if not self.multi_step_loss:
            # Evaluate on the test set for this episode
            outer_loss = model(test)["loss"]
            outer_loss.backward()
            return outer_loss.detach()
            
        else:
            return outer_loss


    def prediction_step(self, model, inputs):
        
        model.eval()
        test = {"input_ids" : inputs["test_input_ids"], "labels" : inputs["test_labels"]}

        # Evaluate on the test set for this episode
        outer_loss = model(test)["loss"].detach()

        return outer_loss




# Train simply on a single task within a meta dataset
def simple_train_model(model, dataset, lr=0.1, epochs=1, vary_train_batch_size=False):

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    model.zero_grad()

    model.train()
    for epoch in range(epochs):
        for mini_batch in dataset["train_batches"]:
            opt.zero_grad()
            if vary_train_batch_size:
                train_loss = model(mini_batch, reduction="sum")["loss"]
            else:
                train_loss = model(mini_batch, reduction="mean")["loss"]
            train_loss.backward()
            opt.step()
            opt.zero_grad()

    model_to_return = model

    return model_to_return

# Train simply on a single task within a meta dataset
def simple_train_model_fig3(model, dataset, lr=0.1, epochs=1, vary_train_batch_size=False):

    output_arr = []

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    model.zero_grad()

    model.train()
    for epoch in range(epochs):
        for mini_batch in dataset["train_batches"]:
            opt.zero_grad()
            if vary_train_batch_size:
                train_loss = model(mini_batch, reduction="sum")["loss"]
            else:
                train_loss = model(mini_batch, reduction="mean")["loss"]
            train_loss.backward()
            opt.step()
            opt.zero_grad()
            
            output_arr.append(model(mini_batch)["probs"].detach().numpy())

    model_to_return = model

    return model_to_return, output_arr







