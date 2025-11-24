# =============================================
# ♟️ AI Self-Playing Chess using CNN Model
# =============================================
import torch
import chess
import pickle
from pathlib import Path
from chess_cnn import ChessCNN

# === Load model & move dictionary ===
base_dir = Path(__file__).resolve().parent
model_path = base_dir / "model" / "chess_model_v4.pt"
dict_path = base_dir / "model" / "move_to_idx.pkl"
##
# Load move_to_idx dictionary
with open(dict_path, "rb") as f:
    move_to_idx = pickle.load(f)

print(f"✅ Loaded move_to_idx with {len(move_to_idx)} moves")

# Load CNN model
model = ChessCNN(num_outputs=len(move_to_idx))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
print(f"✅ Model loaded successfully from {model_path}\n")

# === Start AI self-play ===
print("♟️ Starting AI self-play...\n")
board = chess.Board()

for turn in range(10):  # play 10 moves
    # Convert board to tensor
    tensor = torch.zeros((1, 12, 8, 8))
    piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
                 'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}
    for sq, pc in board.piece_map().items():
        x, y = divmod(sq, 8)
        tensor[0, piece_map[pc.symbol()], x, y] = 1

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        sorted_idx = torch.argsort(probs, dim=1, descending=True)[0]

    move_list = list(move_to_idx.keys())

    # ✅ Pick the first legal predicted move
    predicted_move = None
    for idx in sorted_idx:
        move_str = move_list[idx.item()]
        try:
            board.push_san(move_str)
            predicted_move = move_str
            break
        except:
            continue

    if predicted_move:
        print(f"\n🧩 Move {turn+1}: {predicted_move}")
        print(board)
    else:
        print(" No legal move found. Stopping.")
        break

