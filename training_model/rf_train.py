import chess
import pandas as pd
import pickle
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from chess_dataset import ChessDataset, move_to_idx

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "Lichess_clean_2016_80k_games.csv"
MODEL_PATH = BASE_DIR / "model"/"rf_model_v1.pkl"
MOVE_DICT_PATH = BASE_DIR /"model"/ "move_to_idx_v1.pkl"

# ---------------- FEATURES ----------------
def board_to_features(board: chess.Board):
    features = []

    # piece counts
    for color in [chess.WHITE, chess.BLACK]:
        for piece in [
            chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING
        ]:
            features.append(len(board.pieces(piece, color)))

    # side to move
    features.append(int(board.turn))

    # mobility
    features.append(len(list(board.legal_moves)))

    return np.array(features, dtype=np.float32)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
dataset = ChessDataset(df, max_games=80000)

X, y = [], []

for board_tensor, move_idx in dataset:
    board = chess.Board()
    X.append(board_to_features(board))
    y.append(move_idx)

X = np.array(X)
y = np.array(y)

# ---------------- TRAIN ----------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

print("Validation accuracy:", model.score(X_val, y_val))

# ---------------- SAVE ----------------
joblib.dump(model, MODEL_PATH)
with open(MOVE_DICT_PATH, "wb") as f:
    pickle.dump(move_to_idx, f)

print("✅ Random Forest model saved")
