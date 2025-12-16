import torch
import chess
from torch.utils.data import Dataset

move_to_idx = {}
idx_to_move = {}

def get_move_index(move):
    """Convert move to UCI (e.g. e2e4) and map to unique index."""
    uci = move if isinstance(move, str) else move.uci()
    if uci not in move_to_idx:
        idx = len(move_to_idx)
        move_to_idx[uci] = idx
        idx_to_move[idx] = uci
    return move_to_idx[uci]

class ChessDataset(Dataset):
    """Dataset for predicting next UCI move from board position."""
    def __init__(self, df, max_games=80000, max_samples=80000):
        self.samples = []
        for i, row in enumerate(df.itertuples()):
            board = chess.Board()
            moves = str(row.Moves).split()
            for j in range(len(moves) - 1):
                try:
                    move = board.parse_san(moves[j])
                    board.push(move)
                    next_move = board.parse_san(moves[j + 1])
                    board_tensor = self.board_to_tensor(board)
                    move_idx = get_move_index(next_move.uci())
                    self.samples.append((board_tensor, move_idx))
                    if len(self.samples) >= max_samples:
                        break
                except Exception:
                    break
            if i % 200 == 0:
                print(f"{i}/{max_games} games processed...")
            if i >= max_games:
                break

    def board_to_tensor(self, board):
        tensor = torch.zeros((13, 8, 8), dtype=torch.float32)
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        for sq, pc in board.piece_map().items():
            x, y = divmod(sq, 8)
            tensor[piece_map[pc.symbol()], x, y] = 1
        tensor[12, :, :] = 1.0 if board.turn else 0.0  # side to move
        return tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



