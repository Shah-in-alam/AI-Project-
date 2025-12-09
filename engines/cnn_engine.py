import sys
import torch
import chess
import pickle
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training_model.chess_cnn import ChessCNN

ROOT = ROOT_DIR / "training_model" / "model"
MODEL_PATH = ROOT / "chess_model_v4.pt"
MOVE_DICT_PATH = ROOT / "move_to_idx.pkl"

device = torch.device("cpu")

with open(MOVE_DICT_PATH, "rb") as f:
    move_to_idx = pickle.load(f)

num_outputs = len(move_to_idx)
idx_to_move = {v: k for k, v in move_to_idx.items()}

print(f" Loaded move dictionary with {num_outputs} moves", file=sys.stderr)

model = ChessCNN(num_outputs=num_outputs)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(" Model successfully loaded", file=sys.stderr)

def choose_move(board: chess.Board):
    tensor = torch.zeros((1, 12, 8, 8))
    piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
                 'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}

    for sq, piece in board.piece_map().items():
        x, y = divmod(sq, 8)
        tensor[0, piece_map[piece.symbol()], x, y] = 1

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)

    sorted_idx = torch.argsort(probs, dim=1, descending=True)[0]
    legal = list(board.legal_moves)

    for idx in sorted_idx:
        move_str = idx_to_move.get(idx.item(), "").strip().lower()
        for mv in legal:
            if mv.uci().lower() == move_str:
                print(" CNN move:", move_str)
                return mv

    print(" CNN failed → using first legal move")
    return legal[0] if legal else None

