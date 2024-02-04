import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt

def generate_chess_img(boards, epoch):
    def plot_chessboard(ax):
        chessboard = np.zeros((8, 8))
        chessboard[1::2, 0::2] = 1
        chessboard[0::2, 1::2] = 1
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#EBECD0", "#779556"])
        ax.imshow(chessboard, cmap=cmap, interpolation='none', extent=(0, 8, 0, 8))

    def add_piece(ax, img_path, position):
        piece_img = plt.imread(img_path)
        row, col = position
        extent = [col, col + 1, 7 - row, 8 - row]
        ax.imshow(piece_img, extent=extent, zorder=10)

    fig, axs = plt.subplots(3, 3, figsize=(6, 6))
    piece_img_dir = "./data/pieces/"
    piece_to_img = {1: "pawn_white.png", 2: "knight_white.png", 3: "bishop_white.png", 4: "rook_white.png",
                    5: "queen_white.png", 6: "king_white.png", -1: "pawn_black.png", -2: "knight_black.png",
                    -3: "bishop_black.png", -4: "rook_black.png", -5: "queen_black.png", -6: "king_black.png"}
    game_phases = ["Opening", "Middlegame", "Endgame"]

    for ax, (board, game_phase) in zip(axs.flat, boards):
        plot_chessboard(ax)
        board = np.round_(np.array(board))
        board = np.clip(board, -6, 6).astype(dtype=int)

        for i in range(len(board)):
            for j in range(len(board[0])):
                piece = board[i][j]
                if piece == 0:
                    continue
                piece = piece_to_img[piece]
                add_piece(ax, f"{piece_img_dir}/{piece}", (i, j))

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_title(game_phases[game_phase.tolist().index(1)])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"./results/recent/generations_{epoch}.png")
    # plt.show()


def convert_multichannel_to_single(board):
    piece_channels = list(range(-6, 7))
    piece_channels.remove(0)

    single_channel_board = board.copy()
    for i, channel in enumerate(piece_channels):
        single_channel_board[i] *= channel
    return single_channel_board[:-1].sum(axis=0)
