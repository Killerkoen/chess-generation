import matplotlib.colors
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision.utils import save_image

from .models import Generator, Discriminator
from utils import generate_chess_img, convert_multichannel_to_single


def train(dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator = Generator()
    discriminator = Discriminator()

    adversarial_loss = torch.nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    validity_weight = 1.
    n_epochs = 200

    for epoch in range(n_epochs):
        generator_loss, discriminator_loss = 0.0, 0.0

        for i, (boards, conditions) in enumerate(dataloader):  # dataloader to be implemented
            valid = torch.ones((boards.size(0), 1), device=device, requires_grad=False)
            fake = torch.zeros((boards.size(0), 1), device=device, requires_grad=False)

            real_boards = boards.type(torch.FloatTensor).to(device)
            conditions = conditions.type(torch.FloatTensor).to(device)  # Ensure conditions match expected type

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_boards, conditions), valid)
            noise = torch.randn(boards.size(0), 100, device=device)
            gen_boards = generator(noise, conditions)
            fake_loss = adversarial_loss(discriminator(gen_boards.detach(), conditions), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            discriminator_loss += d_loss.item()

            optimizer_G.zero_grad()
            g_loss = adversarial_loss(discriminator(gen_boards, conditions), valid)
            validity_loss = calculate_validity_loss_multichannel(gen_boards.cpu().data.numpy())
            g_loss = g_loss + validity_weight * validity_loss

            g_loss.backward()
            optimizer_G.step()
            generator_loss += g_loss.item()

        avg_discriminator_loss = discriminator_loss / len(dataloader)
        avg_generator_loss = generator_loss / len(dataloader)
        print(
            f"[Epoch {epoch + 1}/{n_epochs}] [D loss: {avg_discriminator_loss:.4f}] [G loss: {avg_generator_loss:.4f}]")

        if epoch % 1 == 0:
            with torch.no_grad():
                test_noise = torch.randn(9, 100, device=device)
                test_conditions = np.array([np.eye(1, 3, k=k) for k in [0, 1, 2] * 3]).squeeze(1)
                generated_boards = generator(test_noise, torch.FloatTensor(test_conditions).to(device)).cpu().numpy()
                generated_boards = [np.clip(np.round_(board), 0, 1) for board in generated_boards]
                generated_boards = np.array([convert_multichannel_to_single(board) for board in generated_boards])
                generate_chess_img(zip(generated_boards, test_conditions), epoch=epoch)
                print(f"Generated boards at epoch {epoch}: {generated_boards.shape}")


def calculate_validity_loss_multichannel(gen_boards):
    invalidity_scores = np.zeros(gen_boards.shape[0])
    for i, board in enumerate(gen_boards):
        score = 0
        if board[0].sum() != 1 or board[11].sum() != 1:
            score += 1
        if board[1].sum() > 1 or board[10].sum() > 1:
            score += 1
        if board[2].sum() > 2 or board[9].sum() > 2:
            score += 1
        if board[3].sum() > 2 or board[8].sum() > 2:
            score += 1
        if board[4].sum() > 2 or board[7].sum() > 2:
            score += 1
        if board[5].sum() > 8 or board[6].sum() > 8:
            score += 1
        invalidity_scores[i] = score
    validity_loss = invalidity_scores.mean()
    return validity_loss


def calculate_validity_loss_single_channel(gen_boards):
    invalidity_scores = np.zeros(gen_boards.shape[0])
    for i, board in enumerate(gen_boards):
        board = np.round_(np.array(board) * 6)
        board = np.clip(board, -6, 6).astype(dtype=int)
        unique, counts = np.unique(board, return_counts=True)
        piece_frequency = dict(zip(unique, counts))
        score = 0
        for piece, freq in piece_frequency.items():
            if piece == -6 and freq > 1 or piece == 6 and freq > 1:
                score += 1
            elif piece == -5 and freq > 1 or piece == 5 and freq > 1:
                score += 1
            elif piece == -4 and freq > 2 or piece == 4 and freq > 2:
                score += 1
            elif piece == -3 and freq > 2 or piece == 3 and freq > 2:
                score += 1
            elif piece == -2 and freq > 2 or piece == 2 and freq > 2:
                score += 1
            elif piece == -3 and freq > 2 or piece == 3 and freq > 2:
                score += 1
            elif piece == -1 and freq > 8 or piece == 1 and freq > 8:
                score += 1
        invalidity_scores[i] = score
        validity_loss = invalidity_scores.mean()
        return validity_loss
