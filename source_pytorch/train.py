import argparse
import json
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

# imports the model in model.py by name
from model import BinaryClassifier


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(model_info['input_features'],
                             model_info['hidden_dim'],
                             model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# Gets training data in batches from the train.csv file
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"),
                             header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

# Provided training function
def train(model, train_loader, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script.
    The parameters passed are as follows:

    Arguments:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training.
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).

    Return:
    :return the best model's state_dict
    """

    min_loss_value = 0.3
    min_loss_epoch = -1
    model_state_dict = None

    # training loop is provided
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        total_loss = 0

        for batch in train_loader:
            # get data
            batch_x, batch_y = batch
            batch_y.unsqueeze_(0)

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # get predictions from model
            y_pred = model(batch_x)

            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()

        loss = total_loss / len(train_loader)
        print("Epoch: {}, Loss: {}".
              format(epoch, loss))

        if loss < min_loss_value:
            min_loss_value = loss
            min_loss_epoch = epoch
            model_state_dict = model.cpu().state_dict()
            model.to(device)

    print('\nMinimum loss: {}; reached in epoch {}' \
          .format(min_loss_value, min_loss_epoch))

    return model_state_dict

## DONE: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job

    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for
    # training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])

    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    ## DONE: Add args for the three model parameters:
    # input_features, hidden_dim, output_dim
    # Model Parameters
    parser.add_argument('--input-features', type=int, default=2, metavar='IN',
                        help='number of input features')
    parser.add_argument('--hidden-dim', metavar='H', nargs='*', default=[64,64],
                        help='size of hidden layers')
    parser.add_argument('--output-dim', type=int, default=1, metavar='OUT',
                        help='number of outputs')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='D',
                        help='probability of an element to be zeroed')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)


    ## --- Your code here --- ##

    ## DONE:  Build the model by passing in the input params
    # To get params from the parser, call args.argument_name,
    # eg. args.epochs or args.hidden_dim
    # Don't forget to move your model .to(device)
    # to move to GPU, if appropriate
    model = BinaryClassifier(args.input_features, args.hidden_dim,
                             args.output_dim, args.dropout)
    model.to(device)

    ## DONE: Define an optimizer and loss function for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Trains the model
    # (given line of code, which calls the above training function)
    model_state_dict = train(model, train_loader, args.epochs,
                             criterion, optimizer, device)

    ## DONE: complete in the model_info by adding three argument names,
    # Keep the keys of this dictionary as they are
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim
        }
        torch.save(model_info, f)

    ## --- End of your code  --- ##


    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model_state_dict, f)
