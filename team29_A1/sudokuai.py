#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
import copy

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI for Assignment 2.
    Uses:
      - generate_legal_moves()    (rule / legality layer)
      - apply_move_copy()         (game state simulation layer)
      - evaluate_state()          (evaluation layer)
      - alpha_beta()              (search layer, with iterative deepening)
      - compute_best_move()       (action layer)
    """

    def __init__(self):
        super().__init__()

    # ----------------------------------------------------------------------
    # Top-level entry point called by the framework
    # ----------------------------------------------------------------------
    def compute_best_move(self, game_state: GameState) -> None:
        board = game_state.board
        N = board.N
        root_player = game_state.current_player  # 1 or 2

        # ---------------- C0 helpers: uniqueness on a given board ----------------

        def row_has_value(b: SudokuBoard, i: int, value: int) -> bool:
            for col in range(N):
                cell = b.get((i, col))
                if cell != SudokuBoard.empty and cell == value:
                    return True
            return False

        def col_has_value(b: SudokuBoard, j: int, value: int) -> bool:
            for row in range(N):
                cell = b.get((row, j))
                if cell != SudokuBoard.empty and cell == value:
                    return True
            return False

        def block_has_value(b: SudokuBoard, i: int, j: int, value: int) -> bool:
            m = b.region_height()
            n = b.region_width()
            bi = (i // m) * m
            bj = (j // n) * n
            for r in range(bi, bi + m):
                for c in range(bj, bj + n):
                    cell = b.get((r, c))
                    if cell != SudokuBoard.empty and cell == value:
                        return True
            return False

        # ---------------- legality on a given GameState ------------------

        def is_legal_move_state(gs: GameState, i: int, j: int, value: int) -> bool:
            """
            A1-style legality check on an arbitrary GameState gs:
            - square in player_squares()
            - square empty
            - value in [1..N]
            - not in taboo_moves
            - respects C0 (row/column/block uniqueness)
            """
            board_local = gs.board
            N_local = board_local.N
            taboo_local = gs.taboo_moves

            # Allowed squares for the *current* player in this state
            allowed_local = gs.player_squares()
            if allowed_local is None:
                allowed_local = [(r, c) for r in range(N_local) for c in range(N_local)]

            # Must be in allowed squares
            if (i, j) not in allowed_local:
                return False

            # Must be empty
            if board_local.get((i, j)) != SudokuBoard.empty:
                return False

            # Value range
            if not (1 <= value <= N_local):
                return False

            # Not a taboo move
            for t in taboo_local:
                if t.square == (i, j) and t.value == value:
                    return False

            # C0: uniqueness in row / column / block
            if row_has_value(board_local, i, value):
                return False
            if col_has_value(board_local, j, value):
                return False
            if block_has_value(board_local, i, j, value):
                return False

            return True

        # ---------------- generate all legal moves in a state ------------------

        def generate_legal_moves(gs: GameState):
            board_local = gs.board
            N_local = board_local.N

            allowed_local = gs.player_squares()
            if allowed_local is None:
                allowed_local = [(r, c) for r in range(N_local) for c in range(N_local)]

            moves = []
            for (i, j) in allowed_local:
                if board_local.get((i, j)) != SudokuBoard.empty:
                    continue
                for value in range(1, N_local + 1):
                    if is_legal_move_state(gs, i, j, value):
                        moves.append((i, j, value))
            return moves

        # ---------------- move reward approximation (0/1/3/7) ------------------

        def move_reward(board_local: SudokuBoard, i: int, j: int, value: int) -> int:
            """
            Approximate reward for putting `value` at (i,j) on board_local.
            We don't use the solver; instead we check how many regions become full
            (no empty cells) after this move: 0, 1, 2 or 3, mapped to 0, 1, 3, 7.
            """

            N_local = board_local.N

            def row_completed() -> bool:
                for c in range(N_local):
                    if c == j:
                        # we are placing a value here, so treat as filled
                        continue
                    if board_local.get((i, c)) == SudokuBoard.empty:
                        return False
                return True

            def col_completed() -> bool:
                for r in range(N_local):
                    if r == i:
                        continue
                    if board_local.get((r, j)) == SudokuBoard.empty:
                        return False
                return True

            def block_completed() -> bool:
                m = board_local.region_height()
                n = board_local.region_width()
                bi = (i // m) * m
                bj = (j // n) * n
                for r in range(bi, bi + m):
                    for c in range(bj, bj + n):
                        if r == i and c == j:
                            continue
                        if board_local.get((r, c)) == SudokuBoard.empty:
                            return False
                return True

            completed = 0
            if row_completed():
                completed += 1
            if col_completed():
                completed += 1
            if block_completed():
                completed += 1

            if completed == 0:
                return 0
            elif completed == 1:
                return 1
            elif completed == 2:
                return 3
            else:
                return 7

        # ---------------- move ordering heuristic ------------------

        def move_heuristic(gs: GameState, mv) -> int:
            """
            Simple heuristic for move ordering in alpha-beta:
            - prefer moves with higher local reward (more regions completed)
            - then prefer moves in denser rows/cols/blocks
            """
            (i, j, v) = mv
            b = gs.board
            N_local = b.N

            score = 0

            # Row density
            for c in range(N_local):
                if c == j:
                    continue
                if b.get((i, c)) != SudokuBoard.empty:
                    score += 1

            # Column density
            for r in range(N_local):
                if r == i:
                    continue
                if b.get((r, j)) != SudokuBoard.empty:
                    score += 1

            # Block density
            m = b.region_height()
            n = b.region_width()
            bi = (i // m) * m
            bj = (j // n) * n
            for r in range(bi, bi + m):
                for c in range(bj, bj + n):
                    if r == i and c == j:
                        continue
                    if b.get((r, c)) != SudokuBoard.empty:
                        score += 1

            # Reward is much more important
            score += 15 * move_reward(b, i, j, v)

            return score

        # ---------------- game state simulation: apply move on copy ------------------

        def apply_move_copy(gs: GameState, mv):
            """
            Returns a deep-copied child state after applying move mv = (i,j,value):
            - updates board
            - appends to moves list
            - approximates reward and updates scores
            - updates occupied_squares{1,2} (for non-classic rules)
            - switches current_player
            """
            (ri, rj, rv) = mv
            child = copy.deepcopy(gs)

            # Reward computed on pre-move board
            reward = move_reward(child.board, ri, rj, rv)

            # Apply move on the board
            child.board.put((ri, rj), rv)
            child.moves.append(Move((ri, rj), rv))

            # Update score for the player who made this move
            p = gs.current_player  # 1 or 2
            child.scores[p - 1] += reward

            # Update occupied squares (affects player_squares in non-classic modes)
            if child.occupied_squares1 is not None:
                if p == 1:
                    child.occupied_squares1.append((ri, rj))
                else:
                    child.occupied_squares2.append((ri, rj))

            # Switch player
            child.current_player = 3 - p

            return child

        # ---------------- evaluation: always from root_player's POV ------------------

        def evaluate_state(gs: GameState) -> float:
            """
            Evaluate the position from root_player's point of view:
            - Current score difference
            - Max reward I can get on next move
            - Max reward opponent can get on next move (penalized)
            - Small bonus for number of filled cells (progress)

            - Region completion threats/opportunities
            - Simple flexibility (amount of legal moves possible later)
            """
            # 1) Score difference
            score_diff = gs.scores[root_player - 1] - gs.scores[2 - root_player]

            # 2) Best immediate rewards for both players
            def best_immediate_reward_for(player: int) -> int:
                old_player = gs.current_player
                gs.current_player = player
                moves_p = generate_legal_moves(gs)
                best_r = 0
                for (ii, jj, vv) in moves_p:
                    r = move_reward(gs.board, ii, jj, vv)
                    if r > best_r:
                        best_r = r
                gs.current_player = old_player
                return best_r

            my_best  = best_immediate_reward_for(root_player)
            opp_best = best_immediate_reward_for(3 - root_player)

            # 3) Number of filled cells
            filled = 0
            b = gs.board
            for r in range(N):
                for c in range(N):
                    if b.get((r, c)) != SudokuBoard.empty:
                        filled += 1

            # 4) Region completion threats/opportunities (A region (row/col/block) is a threat/opportunity if it has
            # *one empty cell left* and that empty cell belongs to a player's region.)
            
            def region_of(i, j):
                m = b.region_height()
                n = b.region_width()
                bi = (i // m) * m
                bj = (j // n) * n
                return [(r, c) for r in range(bi, bi + m) for c in range(bj, bj + n)]

            threat_score = 0.0
            opportunity_score = 0.0

            # Get allowed zones
            if root_player == 1:
                my_allowed_sq  = gs.allowed_squares1
                opp_allowed_sq = gs.allowed_squares2
            else:
                my_allowed_sq  = gs.allowed_squares2
                opp_allowed_sq = gs.allowed_squares1

            # We gather all squares for both players (not only current player)
            my_squares = gs.occupied_squares1 if root_player == 1 else gs.occupied_squares2
            opp_squares = gs.occupied_squares1 if root_player == 2 else gs.occupied_squares2
            if my_squares is None: my_squares = []
            if opp_squares is None: opp_squares = []

            # Check only squares in allowed regions for both players
            # This is much cheaper than scanning the board.
            def check_region_threat(square_list, is_me: bool):
                nonlocal threat_score, opportunity_score
                for (i, j) in square_list:
                    if b.get((i, j)) != SudokuBoard.empty:
                        continue

                    # Check row, col, and block regions
                    regions = [
                        [(i, cc) for cc in range(N)],                    # row
                        [(rr, j) for rr in range(N)],                    # column
                        region_of(i, j)                                   # block
                    ]

                    for region in regions:
                        empty_cells = [(r, c) for (r, c) in region if b.get((r, c)) == SudokuBoard.empty]
                        if len(empty_cells) == 1:
                            # One move left to complete this region
                            if is_me:
                                opportunity_score += 6.0
                            else:
                                threat_score -= 7.0

            # For my threats/opportunities, use my allowed squares
            if my_allowed_sq is None:
                my_allowed_sq = [(r, c) for r in range(N) for c in range(N)]
            if opp_allowed_sq is None:
                opp_allowed_sq = [(r, c) for r in range(N) for c in range(N)]

            # If squares cannot be determined from current player, fallback:
            if my_allowed_sq is None:
                my_allowed_sq = [(r, c) for r in range(N) for c in range(N)]
            if opp_allowed_sq is None:
                opp_allowed_sq = [(r, c) for r in range(N) for c in range(N)]

            check_region_threat(my_allowed_sq, True)
            check_region_threat(opp_allowed_sq, False)

            # 5) mobility / flexibility: number of legal moves possible later
            def quick_flex(player: int) -> int:
                old_player = gs.current_player
                gs.current_player = player
                moves_p = generate_legal_moves(gs)
                gs.current_player = old_player
                # Only count unique squares, not full multi-value branching
                sq = {(i, j) for (i, j, v) in moves_p}
                # More squares = more flexibility
                return len(sq)

            my_flex  = quick_flex(root_player)
            opp_flex = quick_flex(3 - root_player)

            # Final weighted evaluation
            return (
                2.0 * score_diff +
                1.2 * my_best -
                1.5 * opp_best +
                0.01 * filled +
                opportunity_score +
                threat_score +
                0.05 * (my_flex - opp_flex)
            )


        # ---------------- alpha-beta search with minimax view ------------------

        TIME_LIMIT = 0.45  # seconds (leave a bit of margin for 0.5s total)
        start_time = time.time()

        def alpha_beta(gs: GameState, depth: int, alpha: float, beta: float,
                       maximizing: bool):
            """
            Standard alpha-beta search.
            maximizing == True  iff  gs.current_player == root_player
            """
            if time.time() - start_time > TIME_LIMIT:
                raise TimeoutError

            moves = generate_legal_moves(gs)

            # Leaf node or no moves
            if depth == 0 or not moves:
                return evaluate_state(gs), None

            # ---- move ordering + limit branching factor ----
            moves.sort(key=lambda mv: move_heuristic(gs, mv), reverse=maximizing)

            MAX_MOVES_PER_NODE = 8   
            if len(moves) > MAX_MOVES_PER_NODE:
                moves = moves[:MAX_MOVES_PER_NODE]


            best_move = None

            if maximizing:
                value = -float("inf")
                for mv in moves:
                    child = apply_move_copy(gs, mv)
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
                for mv in moves:
                    child = apply_move_copy(gs, mv)
                    score, _ = alpha_beta(child, depth - 1, alpha, beta, True)
                    if score < value:
                        value = score
                        best_move = mv
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                return value, best_move

        # ---------------- Root: get all legal moves and do iterative deepening ------------------

        legal_moves = generate_legal_moves(game_state)
        if not legal_moves:
            return  # no move to propose

        # Fallback: random legal move (in case search does not finish)
        best_move = random.choice(legal_moves)
        self.propose_move(Move((best_move[0], best_move[1]), best_move[2]))

        MAX_DEPTH = 4

        depth = 1
        while depth <= MAX_DEPTH:
            try:
                if time.time() - start_time > TIME_LIMIT:
                    break
                maximizing_root = (game_state.current_player == root_player)
                score, mv = alpha_beta(game_state, depth,
                                       -float("inf"), float("inf"),
                                       maximizing_root)
                if mv is not None:
                    best_move = mv
                    self.propose_move(Move((mv[0], mv[1]), mv[2]))
            except TimeoutError:
                break
            depth += 1

        # ---------------- Keep proposing the final best move until killed ------------------

        final_move = Move((best_move[0], best_move[1]), best_move[2])
        while True:
            time.sleep(0.1)
            self.propose_move(final_move)
