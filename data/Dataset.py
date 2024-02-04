import re

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import chess
from tqdm import tqdm


class ChessPuzzleDataset(Dataset):
    def __init__(self, csv_file, n_rows=None, multichannel=False, n_instances_per_class=10):
        raw_data = pd.read_csv(csv_file, nrows=n_rows)
        puzzles = raw_data.apply(lambda x: self.__convert_fen_to_board(x["FEN"]), axis=1).tolist()
        game_phases = raw_data.apply(lambda x: re.search(r"opening|middlegame|endgame", x["Themes"]).group(), axis=1).tolist()

        self.__data = []
        game_phase_count = {"opening": 0, "middlegame": 0, "endgame": 0}
        for puzzle, game_phase in zip(puzzles, game_phases):
            if game_phase_count[game_phase] < n_instances_per_class:
                self.__data.append((puzzle, game_phase))
                game_phase_count[game_phase] += 1

        print(f"Class frequency distribution: {game_phase_count}")

        self.__game_phase_indices = {"opening": 0, "middlegame": 1, "endgame": 2}
        self.__multichannel = multichannel

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, i):
        board, game_phase = self.__data[i]
        board = np.array(board) / 6
        game_phase = np.eye(1, 3, k=self.__game_phase_indices[game_phase])[0].tolist()
        if self.__multichannel is True:
            board = self.__convert_to_multichannel(board * 6)

        return np.array(board), np.array(game_phase)

    def __convert_to_multichannel(self, board):
        board = np.array(board)
        piece_channels = list(range(-6, 7))
        piece_channels.remove(0)
        multichannel_board = []

        for channel in piece_channels:
            board_channel = board.copy().astype(float)
            board_channel[board_channel != channel] = 0
            board_channel /= channel
            multichannel_board.append(board_channel.astype(int).tolist())

        empty_board_channel = board.copy().astype(float)
        empty_board_channel[empty_board_channel == 0] = 7
        empty_board_channel[empty_board_channel < 7] = 0
        empty_board_channel /= 7
        multichannel_board.append(empty_board_channel.astype(int).tolist())
        return multichannel_board

    def __convert_fen_to_board(self, fen_string, moves=None):
        board = chess.Board(fen_string)
        if moves is not None:
            board.push(chess.Move.from_uci(moves.split(" ")[0]))
        color = fen_string.split(" ")[1]
        if color == "w":
            board = board.transform(chess.flip_vertical)
            board = board.transform(chess.flip_horizontal)
        board = board.__str__().replace(" ", "").replace("\n", "")
        letter_to_number = {"p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,
                            "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, ".": 0}
        board = [[letter_to_number[board[i][j]] for j in range(len(board[0]))] for i in range(len(board))]
        board = np.array(board).reshape((8, 8))
        return board.tolist()
