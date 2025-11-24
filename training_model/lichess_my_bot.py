# training_model/lichess_my_bot.py
import os
import time
import sys
import chess
import torch
import pickle
import berserk
from pathlib import Path
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


try:
    from engines.cnn_engine import choose_move as engine_choose_move
    print("✅ Using choose_move() from engines.my_cnn_engine")
except Exception as e:
    engine_choose_move = None
    


load_dotenv()
API_TOKEN = os.getenv("LICHESS_BOT_TOKEN")
if not API_TOKEN:
    print(" LICHESS BOT TOKEN NOT FOUND!")
    exit()


if engine_choose_move is None:
    from chess_cnn import ChessCNN

    ROOT = Path(__file__).parent / "model"
    MODEL_PATH = ROOT / "chess_model_v4.pt"
    MOVE_DICT_PATH = ROOT / "move_to_idx.pkl"

    with open(MOVE_DICT_PATH, "rb") as f:
        move_to_idx = pickle.load(f)
    idx_to_move = {v: k for k, v in move_to_idx.items()}

    device = torch.device("cpu")
    model = ChessCNN(len(move_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Local model loaded")

    def choose_move(board):
        tensor = torch.zeros((1, 12, 8, 8))
        piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
                     'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}
        for sq, pc in board.piece_map().items():
            x, y = divmod(sq, 8)
            tensor[0, piece_map[pc.symbol()], x, y] = 1

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)

        sorted_idx = torch.argsort(probs, dim=1, descending=True)[0]
        legal = list(board.legal_moves)

        for i in sorted_idx:
            mv_str = idx_to_move.get(i.item())
            if not mv_str:
                continue
            for mv in legal:
                try:
                    if mv.uci() == mv_str or board.san(mv) == mv_str:
                        return mv
                except:
                    pass
        return None

else:
    choose_move = engine_choose_move


session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)
print("✅ Connected to Lichess Bot API")

print("♟ Waiting for game challenge...")

for event in client.bots.stream_incoming_events():

    # ✅ Auto accept challenge
    if event["type"] == "challenge":
        challenge_id = event["challenge"]["id"]
        print(f"⚔️ Challenge received — accepting ({challenge_id})...")
        client.bots.accept_challenge(challenge_id)
        continue

    if event["type"] == "gameStart":
        game_id = event["game"]["id"]
        bot_color = event["game"]["color"]
        print(f"\n🎮 Game started! https://lichess.org/{game_id}  (playing {bot_color})\n")

        board = chess.Board()
        move_count = 0  # 👈 Track total moves

        for state in client.bots.stream_game_state(game_id):
            moves = state.get("moves", "")
            board.reset()

            if moves.strip():
                for m in moves.split():
                    board.push_uci(m)

            move_count = len(moves.split())  # 👈 update move count

            # ✅ CHECK IF GAME IS OVER
            if board.is_game_over():
                print("\n=== GAME OVER ===")
                print("Result:", board.result())

                result = board.result()
                if result == "1-0":
                    print("🏆 WINNER: WHITE")
                elif result == "0-1":
                    print("🏆 WINNER: BLACK")
                else:
                    print("🤝 It's a DRAW")

                print(f"🔢 Total Moves Played: {move_count}")
                print("================\n")
                break  # Stop bot after game ends

            is_bot_turn = (board.turn == chess.WHITE and bot_color == "white") or \
                          (board.turn == chess.BLACK and bot_color == "black")

            if not is_bot_turn:
                continue

            move = choose_move(board)
            if move:
                print(f"🤖 Playing: {move.uci()}")
                client.bots.make_move(game_id, move.uci())
            else:
                print("⚠ No valid move found")

            time.sleep(1)

