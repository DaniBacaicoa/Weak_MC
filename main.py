import os
import torch
import torch.nn as nn
import numpy as np

import argparse
import pickle

from ucimlrepo import fetch_ucirepo

from src.dataset import Data_handling
from src.model import MLP
from src.weakener import Weakener

import utils.losses as losses
from utils.train_test_loop import train_and_evaluate

def main(args):
    reps = args.reps
    epochs = args.epochs
    batch_size = args.batch_size
    dataset = args.dataset
    loss_type = args.loss_type
    corruption = args.corruption
    corruption_rate = args.corruption_rate
    model = args.model


    for rep,e in enumerate(reps):
        