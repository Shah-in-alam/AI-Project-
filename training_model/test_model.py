import torch
import pickle
from pathlib import Path
import chess
from chess_cnn import ChessCNN

# === Path setup ===
try:
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path.cwd()

model_path = base_dir / "model" / "chess_model_v4.pt"
dict_path = base_dir / "model" / "move_to_idx.pkl"

# === Load move_to_idx ===
with open(dict_path, "rb") as f:
    move_to_idx = pickle.load(f)
print(f"✅ Loaded move_to_idx with {len(move_to_idx)} moves")

# === Load model ===
model = ChessCNN(num_outputs=len(move_to_idx))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
print(f"✅ Model loaded successfully from {model_path}")

# === Test on initial board ===
board = chess.Board()
print("\n♟️ Current Board:")
print(board)

# === Convert to tensor ===
tensor = torch.zeros((1, 12, 8, 8))
piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}
for sq, pc in board.piece_map().items():
    x, y = divmod(sq, 8)
    tensor[0, piece_map[pc.symbol()], x, y] = 1

# === Predict next move ===
with torch.no_grad():
    output = model(tensor)
    probs = torch.softmax(output, dim=1)
    top5 = torch.topk(probs, 5)

print("\n🤖 Top 5 Predicted Moves:")
move_list = list(move_to_idx.keys())
for i, (score, idx) in enumerate(zip(top5.values[0], top5.indices[0])):
    if idx < len(move_list):
        print(f"{i+1}. {move_list[idx]} ({score.item()*100:.2f}%)")


