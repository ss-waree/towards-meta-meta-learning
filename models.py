
import math
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, AutoConfig, PreTrainedModel


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


################################################################################################
# Abstract base class for models
################################################################################################

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        # Every model class should implement these so they can do saving/loading
        self.save_dir = None
        self.name = None

        # This does not have to be overridden in every model; just in model
        # classes that act as wrappers for HuggingFace models
        self.model = None

    def forward(self, batch):

        # Input: batch - a dictionary containing any inputs that the model needs
        # Output: another dictionary, containing any outputs that will be needed from the model
        raise NotImplementedError

    def trainable_parameters(self):

        # Yield the model's trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

    def save(self):
        logging.info("Saving model checkpoint to %s", self.save_dir)
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        if not isinstance(self.model, PreTrainedModel):
            state_dict = self.state_dict()
            torch.save(state_dict, os.path.join(self.save_dir, self.name + ".weights"))
        else:
            output_dir = os.path.join(self.save_dir, self.name)
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)

    def load(self, model_name=None):
        
        # Default to loading the best saved weights for this model
        # (if model_name is provided, this default is overridden to
        # instead load a different pretrained model)
        if model_name is None:
            model_name = os.path.join(self.save_dir, self.name)

        logging.info("Loading model checkpoint from %s", model_name)

        if isinstance(self.model, GPT2LMHeadModel):
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        else:
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(model_name + ".weights"))
            else:
                self.load_state_dict(torch.load(model_name + ".weights", map_location=torch.device('cpu')))


################################################################################################
# Classifier
################################################################################################

# Input: 8 - zero vs one as 2 separate indices
class MLPClassifier(Model):

    def __init__(self, n_features=1, hidden_size=None, n_layers=None, dropout=0.5, nonlinearity="ReLU", save_dir=None, model_name=None):
        super(MLPClassifier, self).__init__()

        self.save_dir = save_dir
        self.name = model_name

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.nonlinearity = nonlinearity

        self.drop = nn.Dropout(dropout)

        self.in_layer = nn.Linear(self.n_features, self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size, 2)

        for layer_index in range(self.n_layers):
            setattr(self, "layer" + str(layer_index), nn.Linear(self.hidden_size, self.hidden_size))

        if self.nonlinearity == "ReLU":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "sigmoid":
            self.nonlinearity = nn.Sigmoid()

        # For squeezing the output into [0,1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, reduction="mean"):

        # print(batch)

        hidden = self.in_layer(batch["input_ids"])
        for layer_index in range(self.n_layers):
            hidden = getattr(self, "layer" + str(layer_index))(hidden)
            hidden = self.nonlinearity(hidden)
            hidden = self.drop(hidden)
        output = self.out_layer(hidden)
        probs = nn.Softmax(dim = 1)(output)

        loss = None
        acc = None
        if "labels" in batch:
            loss_fct = nn.CrossEntropyLoss(reduction=reduction)
            # print("_______________________________\n")
            # print("ids", batch["input_ids"])
            # print("labels", batch["labels"])
            # print("labels squeeze", batch["labels"].view(-1))
            # print("output", output)
            # print("probs", probs)
            loss = loss_fct(output, batch["labels"].view(-1))

            # Compute accuracy
            preds = torch.topk(probs,1)[1]
            # print("preds", preds)
            # print("_______________________________\n")
            correct = torch.sum(preds == batch["labels"])
            #print("correct", correct)
            incorrect = torch.sum(preds != batch["labels"])
            #print("incorrect", incorrect)
            acc = correct * 1.0 / (correct + incorrect)
            acc = acc.item()

        return {"probs" : probs, "loss" : loss, "accuracy" : acc}
    
################################################################################################
# Transformer
################################################################################################

class Transformer(Model):

    def __init__(self, n_features=4, hidden_size=None, n_layers=None, dropout=0.5, nonlinearity="ReLU", save_dir=None, model_name=None):
        super(Transformer, self).__init__()

        self.save_dir = save_dir
        self.name = model_name

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.nonlinearity = nonlinearity

        self.drop = nn.Dropout(dropout)

        self.vocab_size = 30

        # self.in_layer = nn.Linear(self.n_features*3, self.hidden_size)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
    
        # self.positional_encoding = PositionalEncoding(hidden_size)
        self.num_heads = 8
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=self.num_heads, batch_first=True), num_layers=n_layers)
        self.out_layer = nn.Linear(self.hidden_size, 2)

    def forward(self, batch, reduction="mean"):

        # print("batch labels", batch['labels'])
        # print("batch labels shape", batch['labels'].shape)

        hidden = self.embedding(batch['input_ids'].long())
        # print('hidden', hidden.shape)
        
        # print("labels 2", batch['labels'] == 2)
        # print("labels 2 shape", (batch['labels'] == 2).shape)
        # print("squeezed", torch.squeeze((batch['labels'] == 2), dim = 1))
        # print("squeezed shape", torch.squeeze((batch['labels'] == 2), dim = 1).shape)

        if "labels" in batch:
            src_key_padding_mask = torch.squeeze((batch['labels'] == 2), dim = 1)
            output = self.transformer(hidden, src_key_padding_mask=src_key_padding_mask)
        else: 
            output = self.transformer(hidden)

        # print(output.shape)

        output = self.out_layer(output)
        # print("before reshape", output, output.shape)
        output = output.reshape((-1, 2))
        # print("after reshape", output, output.shape)

        probs = nn.Softmax(dim=1)(output)
        # print("probs", probs, "\n")
        # probs = probs[:, 1]
        # print("probs collapsed", probs, "\n")

        loss = None
        acc = None
        if "labels" in batch:
            labels = batch["labels"].view(-1)
            # print('LABELS', labels, labels.shape)
            loss_fct = nn.CrossEntropyLoss(reduction=reduction, ignore_index=2)
            loss = loss_fct(output, labels)
            # print('output shape', output.shape)

            # indices = torch.where(labels != 2)
            # probs = probs[indices]
            # labels = labels[indices]

            # Compute accuracy
            preds = (probs[:, 1] > 0.5)
            correct = torch.sum(preds == labels)
            incorrect = torch.sum(preds != labels)
            acc = correct * 1.0 / (correct + incorrect)
            acc = acc.item()
            # correct_array = (preds == labels)
            # print('correct array from model pass', correct_array)

            # if acc != 1:
            #     print('wrong example', batch['input_ids'])
            #     print('preds', preds)
            #     print('true lables', labels)
        #return {"probs" : probs, "loss" : loss, "accuracy" : acc, "correct_array": correct_array}
        return {"probs" : probs, "loss" : loss, "accuracy" : acc}


if __name__ == "__main__":
    inputs = [[0,0,0,0], [0,1,1,0], [0,1,1,0], [1,1,1,0], [1,1,1,0]]
    labels = [0,0,0,1,1]

    batch = {"input_ids" : torch.FloatTensor(inputs), "labels" : torch.FloatTensor(labels).unsqueeze(1)}
    model = MLPClassifier(n_features=4, hidden_size=7, n_layers=3, dropout=0.0, nonlinearity="ReLU")
    print(model(batch))







