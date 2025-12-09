# engines/my_cnn_engine.py
import sys
from pathlib import Path
import pickle
import torch
import torch.nn.functional as F
import chess
from stockfish import Stockfish

# allow importing training_model modules when running from repo root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# import your model class
from training_model.chess_cnn import ChessCNN

# ---------- CONFIG ----------
STOCKFISH_PATH = str(ROOT_DIR / "bin" / "stockfish.exe")  #  real engine path
MODEL_DIR = ROOT_DIR / "training_model" / "model"
MODEL_PATH = MODEL_DIR / "chess_model_v4.pt"
MOVE_DICT_PATH = MODEL_DIR / "move_to_idx.pkl"
DEVICE = torch.device("cpu")

CONF_THRESH = 0.20  # minimum probability to trust CNN move
TOP_K = 30
TEMPERATURE = 1.0
# ----------------------------

# Load move dictionary
with open(MOVE_DICT_PATH, "rb") as f:
    move_to_idx = pickle.load(f)
idx_to_move = {v: k for k, v in move_to_idx.items()}
NUM_OUTPUTS = len(move_to_idx)

# Load CNN model
model = ChessCNN(num_outputs=NUM_OUTPUTS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load real Stockfish engine
stockfish = Stockfish(path=STOCKFISH_PATH, parameters={
    "Threads": 2,
    "Minimum Thinking Time": 30
})

print(" CNN model loaded with", NUM_OUTPUTS, "moves", file=sys.stderr)
print(" Stockfish engine loaded from:", STOCKFISH_PATH, file=sys.stderr)

# -------- helpers --------
def board_to_tensor(board: chess.Board):
    tensor = torch.zeros((1, 12, 8, 8), dtype=torch.float32, device=DEVICE)
    piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
                 'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}
    for sq, pc in board.piece_map().items():
        x, y = divmod(sq, 8)
        tensor[0, piece_map[pc.symbol()], x, y] = 1
    return tensor

# -------- move chooser --------
def choose_move(board: chess.Board):

    # 1️⃣ Try CNN
    try:
        x = board_to_tensor(board)
        with torch.no_grad():
            out = model(x)
            probs = F.softmax(out / TEMPERATURE, dim=1)[0]

        legal_moves = list(board.legal_moves)
        legal_ucis = {m.uci().lower() for m in legal_moves}

        topk = torch.topk(probs, min(TOP_K, len(probs))).indices.tolist()

        for idx in topk:
            pred_move = idx_to_move.get(idx, "").lower()
            if pred_move in legal_ucis:
                print(" CNN move:", pred_move)
                return chess.Move.from_uci(pred_move)

        # if top prediction is HIGH confidence, try harder
        best_idx = topk[0]
        best_move = idx_to_move.get(best_idx, "").lower()
        if float(probs[best_idx]) > CONF_THRESH and best_move in legal_ucis:
            print(" CNN confident move:", best_move)
            return chess.Move.from_uci(best_move)
        else:
            print("CNN failed → trying Stockfish")

    except Exception as e:
        print("⚠ CNN crashed:", e)

    # 2️⃣ Fallback to Stockfish
    try:
        stockfish.set_fen_position(board.fen())
        best = stockfish.get_best_move()
        if best:
            print("♞ Stockfish move:", best)
            return chess.Move.from_uci(best)
    except Exception as e:
        print("⚠ Stockfish error:", e)

    # 3️⃣ Emergency fallback
    print("⚠ Last fallback: random legal move")
    return list(board.legal_moves)[0] if board.legal_moves else None

