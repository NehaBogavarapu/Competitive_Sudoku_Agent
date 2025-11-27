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

        # Return list of all values currently placed in a block
        def block_values(gs, block_row, block_col):
            board = gs.board
            m = board.region_height()
            n = board.region_width()
            vals = []
            for r in range(block_row * m, block_row * m + m):
                for c in range(block_col * n, block_col * n + n):
                    cell = board.get((r, c))
                    if cell != SudokuBoard.empty:
                        vals.append(cell)
            return vals


        # How close is player p to gaining this region?
        # Return: (my_threat, opp_threat)
        def region_threat(gs, block_row, block_col, my_player):
            vals = block_values(gs, block_row, block_col)
            size = len(vals)
            # Threat = how many empty cells remain before completion
            remaining = gs.board.region_height() * gs.board.region_width() - size

            # A region is “controlled” by the player who currently wins it if it ends now
            # Using the scoring logic in the framework:
            # Score = size (used inside scoring for territory)
            # Determine who benefits
            my_score_now = size if my_player == gs.current_player else -size

            # Because control is fuzzy during filling, use threat levels
            my_threat = 0
            opp_threat = 0

            # If region is nearly complete, treat it as a high threat
            if remaining <= 2:
                # If I am currently the player who would win, good
                if my_score_now > 0:
                    my_threat = 3 - remaining   # closer to completion → higher threat
                else:
                    opp_threat = 3 - remaining

            return my_threat, opp_threat


        # Count legal moves for a given player
        def mobility(gs, player):
            board = gs.board
            N = board.N
            allowed = allowed_squares_for(gs, player)

            count = 0
            for (i, j) in allowed:
                if board.get((i, j)) != SudokuBoard.empty:
                    continue
                for v in range(1, N+1):
                    if is_legal_move(i, j, v):
                        count += 1
            return count


        def allowed_squares_for(gs, player):
            # If asking for current player, just use the provided API
            if player == gs.current_player:
                sq = gs.player_squares()
                if sq is None:
                    N = gs.board.N
                    return [(i, j) for i in range(N) for j in range(N)]
                return sq

            # Otherwise: we need to reconstruct the opponent’s allowed squares.
            N = gs.board.N
            board = gs.board

            # In competitive sudoku: player 1 plays top half, player 2 bottom half (for 4x4)
            # More generally:
            region_size = board.N // 2

            if player == 1:
                rows = range(0, region_size)
            else:  # player == 2
                rows = range(region_size, N)

            return [(r, c) for r in rows for c in range(N)]


        # ---------------- EVALUATION FUNCTION -----------------

        def evaluate(gs):
            board = gs.board
            N = board.N
            my_player = game_state.current_player
            opp_player = 3 - my_player

            # -------- 1. Material advantage (raw score difference) ----------
            score_term = gs.scores[my_player - 1] - gs.scores[opp_player - 1]

            # -------- 2. Region control & Threat analysis ----------
            my_threat_sum = 0
            opp_threat_sum = 0

            m = board.region_height()
            n = board.region_width()

            for br in range(N // m):
                for bc in range(N // n):
                    t_my, t_opp = region_threat(gs, br, bc, my_player)
                    my_threat_sum += t_my
                    opp_threat_sum += t_opp

            region_term = 2.0 * (my_threat_sum - opp_threat_sum)

            # -------- 3. Mobility (future flexibility) ----------
            my_moves = mobility(gs, my_player)
            opp_moves = mobility(gs, opp_player)
            mobility_term = 0.5 * (my_moves - opp_moves)

            # -------- 4. Board progress (prefer states where you restrict opponent) ----------
            filled = sum(1 for r in range(N) for c in range(N)
                        if board.get((r, c)) != SudokuBoard.empty)
            progress_term = 0.002 * filled

            # -------- 5. Long-term penalty for enabling opponent ----------
            # If opponent has huge threat advantage, punish the state
            long_term_risk = -1.2 * max(0, opp_threat_sum - my_threat_sum)

            # FINAL SCORE
            return score_term + region_term + mobility_term + progress_term + long_term_risk


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
