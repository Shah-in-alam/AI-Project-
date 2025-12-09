# """
# Some example classes for people who want to create a homemade bot.

# With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
# """
# import chess
# from chess.engine import PlayResult, Limit
# import random
# from lib.engine_wrapper import MinimalEngine
# from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE
# import logging


# # Use this logger variable to print messages to the console or log files.
# # logger.info("message") will always print "message" to the console or log file.
# # logger.debug("message") will only print "message" if verbose logging is enabled.
# logger = logging.getLogger(__name__)


# class ExampleEngine(MinimalEngine):
#     """An example engine that all homemade engines inherit."""


# # Bot names and ideas from tom7's excellent eloWorld video

# class RandomMove(ExampleEngine):
#     """Get a random move."""

#     def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
#         """Choose a random move."""
#         return PlayResult(random.choice(list(board.legal_moves)), None)


# class Alphabetical(ExampleEngine):
#     """Get the first move when sorted by san representation."""

#     def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
#         """Choose the first move alphabetically."""
#         moves = list(board.legal_moves)
#         moves.sort(key=board.san)
#         return PlayResult(moves[0], None)


# class FirstMove(ExampleEngine):
#     """Get the first move when sorted by uci representation."""

#     def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:  # noqa: ARG002
#         """Choose the first move alphabetically in uci representation."""
#         moves = list(board.legal_moves)
#         moves.sort(key=str)
#         return PlayResult(moves[0], None)


# class ComboEngine(ExampleEngine):
#     """
#     Get a move using multiple different methods.

#     This engine demonstrates how one can use `time_limit`, `draw_offered`, and `root_moves`.
#     """

#     def search(self,
#                board: chess.Board,
#                time_limit: Limit,
#                ponder: bool,  # noqa: ARG002
#                draw_offered: bool,
#                root_moves: MOVE) -> PlayResult:
#         """
#         Choose a move using multiple different methods.

#         :param board: The current position.
#         :param time_limit: Conditions for how long the engine can search (e.g. we have 10 seconds and search up to depth 10).
#         :param ponder: Whether the engine can ponder after playing a move.
#         :param draw_offered: Whether the bot was offered a draw.
#         :param root_moves: If it is a list, the engine should only play a move that is in `root_moves`.
#         :return: The move to play.
#         """
#         if isinstance(time_limit.time, int):
#             my_time = time_limit.time
#             my_inc = 0
#         elif board.turn == chess.WHITE:
#             my_time = time_limit.white_clock if isinstance(time_limit.white_clock, int) else 0
#             my_inc = time_limit.white_inc if isinstance(time_limit.white_inc, int) else 0
#         else:
#             my_time = time_limit.black_clock if isinstance(time_limit.black_clock, int) else 0
#             my_inc = time_limit.black_inc if isinstance(time_limit.black_inc, int) else 0

#         possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)

#         if my_time / 60 + my_inc > 10:
#             # Choose a random move.
#             move = random.choice(possible_moves)
#         else:
#             # Choose the first move alphabetically in uci representation.
#             possible_moves.sort(key=str)
#             move = possible_moves[0]
#         return PlayResult(move, None, draw_offered=draw_offered)

"""
Homemade engines for lichess-bot + your CNN engine.
"""
import logging
import random
import chess
from chess.engine import PlayResult, Limit

# Required lichess-bot imports
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import HOMEMADE_ARGS_TYPE, MOVE

logger = logging.getLogger(__name__)

# ============================================================
# ❗ DO NOT REMOVE — lichess-bot requires these classes
# ============================================================

class ExampleEngine(MinimalEngine):
    """Base class for all custom engines used by lichess-bot."""
    pass


class RandomMove(ExampleEngine):
    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE):
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE):
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE):
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)


class ComboEngine(ExampleEngine):
    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,
               draw_offered: bool,
               root_moves: MOVE):

        possible_moves = root_moves if isinstance(root_moves, list) else list(board.legal_moves)
        possible_moves.sort(key=str)
        return PlayResult(possible_moves[0], None)

# ============================================================
# ⭐ YOUR CNN MODEL INTEGRATION
# ============================================================

import torch
import pickle
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "training_model" / "model"

MODEL_PATH = MODEL_DIR / "chess_model_v4.pt"
MOVE_DICT_PATH = MODEL_DIR / "move_to_idx.pkl"

# Load move dictionary
with open(MOVE_DICT_PATH, "rb") as f:
    move_to_idx = pickle.load(f)
idx_to_move = {v: k for k, v in move_to_idx.items()}

# Import your model
from training_model.chess_cnn import ChessCNN

device = torch.device("cpu")
model = ChessCNN(num_outputs=len(move_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

logger.info(f"✅ CNN model loaded with {len(move_to_idx)} moves")

def cnn_choose_move(board: chess.Board):
    """Return the best move predicted by your CNN model."""
    tensor = torch.zeros((1, 12, 8, 8))

    piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
                 'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}

    # Encode board
    for sq, pc in board.piece_map().items():
        x, y = divmod(sq, 8)
        tensor[0, piece_map[pc.symbol()], x, y] = 1

    # Predict
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]

    sorted_idx = torch.argsort(probs, descending=True)
    legal_moves = list(board.legal_moves)

    # Try to match CNN prediction with legal moves
    for idx in sorted_idx:
        mv_str = idx_to_move.get(idx.item())
        if mv_str is None:
            continue

        for mv in legal_moves:
            try:
                if mv.uci() == mv_str or board.san(mv) == mv_str:
                    return mv
            except:
                pass

    # CNN failed — return None
    return None

# ============================================================
#  HYBRID ENGINE CLASS (CNN first, fallback to legal move)
# ============================================================

class MyCNNAI(ExampleEngine):
    """Your final engine: CNN first, fallback to legal move."""

    def search(self,
               board: chess.Board,
               time_limit: Limit,
               ponder: bool,
               draw_offered: bool,
               root_moves: MOVE):

        # root_moves support from lichess-bot
        if isinstance(root_moves, list):
            limited_legal = root_moves
        else:
            limited_legal = list(board.legal_moves)

        # Try CNN
        try:
            mv = cnn_choose_move(board)
        except Exception as e:
            logger.error(f"CNN error: {e}")
            mv = None

        # If CNN move not valid or None, fallback
        if mv not in limited_legal:
            if mv:
                logger.info(f"CNN suggested illegal move: {mv}")
            mv = limited_legal[0]

        logger.info(f"🤖 CNN Engine move: {mv}")
        return PlayResult(mv, None)
