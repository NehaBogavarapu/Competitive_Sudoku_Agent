#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()
    
    def compute_best_move(self, game_state: GameState) -> None:
        board = game_state.board
        N = board.N
        taboo = game_state.taboo_moves

        allowed = game_state.player_squares()
        if allowed is None:
            allowed = [(i, j) for i in range(N) for j in range(N)]

        # ---------- C0 CHECKS (row / column / block uniqueness) ----------

        def row_has_value(i, value):
            for col in range(N):
                cell = board.get((i, col))
                if cell != SudokuBoard.empty and cell == value:
                    return True
            return False

        def col_has_value(j, value):
            for row in range(N):
                cell = board.get((row, j))
                if cell != SudokuBoard.empty and cell == value:
                    return True
            return False

        def block_has_value(i, j, value):
            m = board.region_height()
            n = board.region_width()
            bi = (i // m) * m
            bj = (j // n) * n
            for r in range(bi, bi + m):
                for c in range(bj, bj + n):
                    cell = board.get((r, c))
                    if cell != SudokuBoard.empty and cell == value:
                        return True
            return False

        # ------------------ master legality checker ----------------------

        def is_legal_move(i, j, value):
            if (i, j) not in allowed:
                return False
            if board.get((i, j)) != SudokuBoard.empty:
                return False
            if not (1 <= value <= N):
                return False
            for t in taboo:
                if t.square == (i, j) and t.value == value:
                    return False
            if row_has_value(i, value):
                return False
            if col_has_value(j, value):
                return False
            if block_has_value(i, j, value):
                return False
            return True

        # ------------------ move score (kept for fallback ordering) ----------------------

        def move_score(i, j, value):
            score = 0

            # row
            for col in range(N):
                if col == j:
                    score += 1
                elif board.get((i, col)) != SudokuBoard.empty:
                    score += 1

            # column
            for row in range(N):
                if row == i:
                    continue
                if board.get((row, j)) != SudokuBoard.empty:
                    score += 1

            # block
            m = board.region_height()
            n = board.region_width()
            bi = (i // m) * m
            bj = (j // n) * n
            for r in range(bi, bi + m):
                for c in range(bj, bj + n):
                    if r == i and c == j:
                        score += 1
                    elif board.get((r, c)) != SudokuBoard.empty:
                        score += 1
            return score

        # ---------------- collect all legal moves -----------------

        legal_moves = []
        for (i, j) in allowed:
            if board.get((i, j)) != SudokuBoard.empty:
                continue
            for value in range(1, N + 1):
                if is_legal_move(i, j, value):
                    legal_moves.append((i, j, value))

        if not legal_moves:
            return

        # ---------------- STATE CLONE + APPLY -----------------

        import copy
        def apply_move_to_state(gs, mv):
            (ri, rj, rv) = mv
            child = copy.deepcopy(gs)
            child.board.put((ri, rj), rv)
            child.moves.append(Move((ri, rj), rv))
            child.current_player = 3 - gs.current_player
            return child

        # ---------------- EVALUATION FUNCTION -----------------

        def evaluate(gs):
            filled = sum(1 for r in range(N) for c in range(N)
                        if gs.board.get((r, c)) != SudokuBoard.empty)
            score_diff = gs.scores[game_state.current_player - 1] - gs.scores[2 - game_state.current_player]
            return filled * 0.01 + score_diff

        # ---------------- ALPHA-BETA SEARCH -----------------

        import time
        start_time = time.time()
        TIME_LIMIT = 0.45

        def alpha_beta(gs, depth, alpha, beta, maximizing):
            if time.time() - start_time > TIME_LIMIT:
                raise TimeoutError

            # terminal if no moves
            next_moves = []
            allowed_local = gs.player_squares()
            if allowed_local is None:
                allowed_local = [(i, j) for i in range(N) for j in range(N)]

            for (i, j) in allowed_local:
                if gs.board.get((i, j)) != SudokuBoard.empty:
                    continue
                for v in range(1, N + 1):
                    if is_legal_move(i, j, v):
                        next_moves.append((i, j, v))

            if depth == 0 or not next_moves:
                return evaluate(gs), None

            best_move = None

            if maximizing:
                value = -float("inf")
                # ordered for better pruning
                next_moves.sort(key=lambda mv: move_score(*mv), reverse=True)

                for mv in next_moves:
                    child = apply_move_to_state(gs, mv)
                    score, _ = alpha_beta(child, depth - 1, alpha, beta, False)
                    if score > value:
                        value = score
                        best_move = mv
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value, best_move

            else:
                value = float("inf")
                next_moves.sort(key=lambda mv: move_score(*mv))

                for mv in next_moves:
                    child = apply_move_to_state(gs, mv)
                    score, _ = alpha_beta(child, depth - 1, alpha, beta, True)
                    if score < value:
                        value = score
                        best_move = mv
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                return value, best_move

        # ---------------- ITERATIVE DEEPENING -----------------

        best_move = random.choice(legal_moves)
        self.propose_move(Move((best_move[0], best_move[1]), best_move[2]))

        depth = 1
        while True:
            try:
                if time.time() - start_time > TIME_LIMIT:
                    break
                score, mv = alpha_beta(game_state, depth, -float("inf"), float("inf"), True)
                if mv is not None:
                    best_move = mv
                    self.propose_move(Move((mv[0], mv[1]), mv[2]))
            except TimeoutError:
                break
            depth += 1

        # ---------------- KEEP PROPOSING SAME MOVE -----------------

        final_move = Move((best_move[0], best_move[1]), best_move[2])
        while True:
            time.sleep(0.1)
            self.propose_move(final_move)
