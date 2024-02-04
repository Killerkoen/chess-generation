import numpy as np
from torch.utils.data import DataLoader
from data.Dataset import ChessPuzzleDataset

from GAN.train import train, calculate_validity_loss_single_channel, calculate_validity_loss_multichannel
from utils import generate_chess_img, convert_multichannel_to_single

puzzles = ChessPuzzleDataset("./data/saved/lichess_raw_puzzles.csv", n_rows=10000, multichannel=True,
                             n_instances_per_class=1000)
dataloader = DataLoader(puzzles, batch_size=64, shuffle=True)
train(dataloader)
