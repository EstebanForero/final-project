import numpy as np
import random
import pickle 
import os
import time
from connect4.policy import Policy
from connect4.connect_state import ConnectState



def score_board(board, my_piece):
    enemy_piece = -my_piece
    total = 0.0

    # How valuable is each column? Center = more winning lines
    # Col:       0     1     2     3     4     5     6
    col_value = [0.02, 0.06, 0.12, 0.20, 0.12, 0.06, 0.02]

    def score_window(window):
        # Count my pieces, enemy pieces, and empty spaces in a 4-cell window
        mine  = np.count_nonzero(window == my_piece)
        enemy = np.count_nonzero(window == enemy_piece)
        empty = np.count_nonzero(window == 0)

        # Ignore windows where both players have pieces (blocked)
        if mine > 0 and enemy > 0:
            return 0.0

        if   mine == 3 and empty == 1: return +0.6   
        elif mine == 2 and empty == 2: return +0.2   
        elif enemy == 3 and empty == 1: return -0.7  
        elif enemy == 2 and empty == 2: return -0.3  
        return 0.0


    for row in range(6):
        for col in range(4):
            total += score_window(board[row, col:col+4])          # horizontal →

    for col in range(7):
        for row in range(3):
            total += score_window(board[row:row+4, col])          # vertical ↓

    for row in range(3):
        for col in range(4):
            diag_down = np.array([board[row+i, col+i] for i in range(4)])
            diag_up   = np.array([board[row+3-i, col+i] for i in range(4)])
            total += score_window(diag_down)                      # diagonal ↘
            total += score_window(diag_up)                        # diagonal ↗

    # Bonus for owning pieces in valuable (center) columns
    for col in range(7):
        my_pieces_in_col = np.count_nonzero(board[:, col] == my_piece)
        total += my_pieces_in_col * col_value[col]

    return total


# ================================================================
# PART 2: THE AGENT
# ================================================================
class RolloutAgent2(Policy):

    TIME_LIMIT   = 9.0   # seconds per turn (leave 1s buffer from the 10s budget)
    ALPHA        = 0.15  # how fast Q-values update (learning rate)
    CENTER_COLS  = [3, 2, 4, 1, 5, 0, 6]   # preferred column order
    CENTER_BONUS = np.array([0.02, 0.06, 0.12, 0.20, 0.12, 0.06, 0.02])

    # A state needs this many visits before we trust its Q-values
    TRUST_AFTER = 10

    def __init__(self):
        # Load Q-values learned in previous games
        # Q-TABLE: maps board_state → [7 values, one per column]
        self.q_table = self._load_memory()

    def mount(self, timeout=None):
        # Tournament calls this before each game.
        self._save_memory()
        self.q_table = self._load_memory()

    # ============================================================
    # HELPER: figure out whose turn it is from the board
    # ============================================================
    @staticmethod
    def whose_turn(board):
        # Player 1 always goes first, so equal pieces = player 1's turn
        if np.count_nonzero(board == 1) == np.count_nonzero(board == -1):
            return 1
        return -1

    # ============================================================
    # HELPER: would dropping in this column win for this player?
    # ============================================================
    @staticmethod
    def wins_immediately(board, col, player):
        if board[0, col] != 0:
            return False  # column is full

        # Make a copy and drop the piece
        test = board.copy()
        for row in range(5, -1, -1):
            if test[row, col] == 0:
                test[row, col] = player
                break

        # Check all 4 directions for 4 in a row
        for r in range(6):
            for c in range(7):
                if test[r, c] != player:
                    continue
                if c+3 < 7 and all(test[r, c+i] == player for i in range(4)): return True
                if r+3 < 6 and all(test[r+i, c] == player for i in range(4)): return True
                if r+3 < 6 and c+3 < 7 and all(test[r+i, c+i] == player for i in range(4)): return True
                if r+3 < 6 and c-3 >= 0 and all(test[r+i, c-i] == player for i in range(4)): return True
        return False

    # ============================================================
    # HELPER: how many 3-in-a-row threats does this move create?
    # If >= 2, opponent can't block all of them → guaranteed win
    # ============================================================
    def count_threats_created(self, board, col, player):
        test = board.copy()
        for row in range(5, -1, -1):
            if test[row, col] == 0:
                test[row, col] = player
                break

        threat_count = 0
        enemy = -player

        def check_window(window):
            nonlocal threat_count
            mine  = np.count_nonzero(window == player)
            enemy_in = np.count_nonzero(window == enemy)
            empty = np.count_nonzero(window == 0)
            if mine == 3 and empty == 1 and enemy_in == 0:
                threat_count += 1

        for row in range(6):
            for col2 in range(4):
                check_window(test[row, col2:col2+4])
        for col2 in range(7):
            for row in range(3):
                check_window(test[row:row+4, col2])
        for row in range(3):
            for col2 in range(4):
                check_window(np.array([test[row+i,   col2+i] for i in range(4)]))
                check_window(np.array([test[row+3-i, col2+i] for i in range(4)]))

        return threat_count

    # ============================================================
    # HELPER: does this column have 3 stacked vertically?
    # ============================================================
    def has_vertical_threat(self, board, col, player):
        for row in range(3):
            window = board[row:row+4, col]
            if np.count_nonzero(window == player) == 3 and np.count_nonzero(window == 0) == 1:
                return True
        return False

    # ============================================================
    # MAIN DECISION — called every turn
    # ============================================================
    def act(self, s):
        me    = self.whose_turn(s)
        enemy = -me
        free  = [c for c in range(7) if s[0, c] == 0]

        if not free:
            return 0

        turn_start = time.time()

        def finish(col, reason=""):
            elapsed = time.time() - turn_start
            print(f"Rollout2: {elapsed:.2f}s  col={col}  [{reason}]")
            return col

        # --------------------------------------------------------
        # LAYER 1: INSTANT DECISIONS (heuristic rules, no simulation)
        # These fire before any Q-lookup or simulation.
        # --------------------------------------------------------

        # Rule 1: If I can win right now, do it
        for col in free:
            if self.wins_immediately(s, col, me):
                return finish(col, "win")

        # Rule 2: If enemy wins next move, block it
        for col in free:
            if self.wins_immediately(s, col, enemy):
                return finish(col, "block")

        # Rule 3: If I can create 2+ threats at once (fork), do it
        # Opponent can only block one → I win next turn guaranteed
        for col in free:
            if self.count_threats_created(s, col, me) >= 2:
                return finish(col, "fork")

        # Rule 4: Block enemy fork
        for col in free:
            if self.count_threats_created(s, col, enemy) >= 2:
                return finish(col, "block fork")

        # Rule 5: Block enemy stacking 3 vertically
        for col in free:
            if self.has_vertical_threat(s, col, enemy):
                return finish(col, "block vertical")

        # --------------------------------------------------------
        # LAYER 2: Q-TABLE + SIMULATION DECISION
        # No forcing move found — use learned values or simulate.
        # --------------------------------------------------------


        state_id = s.tobytes()
        if state_id not in self.q_table:
            self.q_table[state_id] = {
                "q":     np.zeros(7),
                "n":     np.zeros(7),
                "total": 0
            }

        entry = self.q_table[state_id]

        # Which columns have been tried enough to trust?
        trusted = [c for c in free if entry["n"][c] >= self.TRUST_AFTER]

        if trusted:
            # ── EXPLOITATION ──────────────────────────────────
            # Q-TABLE READ: acting on what we already learned
            best_col = max(trusted, key=lambda c: entry["q"][c])

            # Keep running simulations on the chosen column until time runs out.
            # This continuously refines the Q-value while we still have budget.
            sims_run = 0
            while time.time() - turn_start < self.TIME_LIMIT:
                result = self.run_one_simulation(s, best_col, me)

                # Q-TABLE WRITE: update Q-value with new simulation result
                # Formula: Q = Q + alpha * (new_result - old_Q)
                # This slowly moves Q toward the true average outcome
                entry["q"][best_col] += self.ALPHA * (result - entry["q"][best_col])
                entry["n"][best_col] += 1
                entry["total"]       += 1
                sims_run += 1

            # Q-TABLE WRITE: save to disk after exploitation turn
            self._save_memory()
            return finish(best_col, f"exploit Q  sims={sims_run}")

        else:
            # ── EXPLORATION ───────────────────────────────────
            # New or under-visited state. Run simulations on every
            # free column and pick the best. This builds initial Q-values.
            #
            # We keep looping through all columns until time runs out,
            # so each column gets roughly equal simulation budget.
            scores     = {col: 0.0 for col in free}
            sim_counts = {col: 0   for col in free}
            col_index  = 0

            # Run simulations round-robin across columns until time is up
            while time.time() - turn_start < self.TIME_LIMIT:
                col = free[col_index % len(free)]
                result = self.run_one_simulation(s, col, me)

                # Q-TABLE WRITE: accumulate result into running average
                entry["q"][col] += self.ALPHA * (result - entry["q"][col])
                entry["n"][col] += 1
                entry["total"]  += 1

                scores[col]     += result
                sim_counts[col] += 1
                col_index       += 1

            # Pick column with best average simulation result
            # Add center bonus to break ties toward strategic columns
            best_col = max(
                free,
                key=lambda c: (
                    (scores[c] / sim_counts[c] if sim_counts[c] > 0 else 0)
                    + self.CENTER_BONUS[c]
                )
            )

            total_sims = sum(sim_counts.values())
            # Q-TABLE WRITE: save to disk after exploration turn
            self._save_memory()
            return finish(best_col, f"explore  sims={total_sims}")

    # ============================================================
    # ONE SIMULATION: play out a full game from (board, col)
    # Returns the outcome from my perspective: +10, -10, or 0
    # ============================================================
    def run_one_simulation(self, board, col, me):
        # Start the simulation by playing the chosen column
        state = ConnectState(board=board, player=me).transition(col)

        # Play random(ish) moves until the game ends
        while not state.is_final():
            state = state.transition(self.simulation_move(state))

        winner = state.get_winner()
        outcome = 10.0 if winner == me else (-10.0 if winner == -me else 0.0)

        # Add board pattern score to the terminal reward (reward shaping)
        # This gives richer signal than just win/loss
        outcome += score_board(state.board, me)

        return outcome

    # ============================================================
    # SIMULATION MOVE POLICY
    # Decides moves during simulations — not purely random:
    #   1. Take an immediate win if available
    #   2. Prefer center columns (45% chance each)
    #   3. Otherwise random
    # ============================================================
    def simulation_move(self, state):
        free   = state.get_free_cols()
        player = state.player

        # Grab any immediate win
        for col in free:
            if state.transition(col).get_winner() == player:
                return col

        # Bias toward center columns
        for col in self.CENTER_COLS:
            if col in free and random.random() < 0.45:
                return col

        return random.choice(free)

    # ============================================================
    # PERSISTENCE: save/load Q-table to disk
    # ============================================================
    def _get_memory_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cerebro.bin")

    def _load_memory(self):
        path = self._get_memory_path()
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                return data
            except Exception as e:
                print(f"[Memory] Load failed: {e}, starting fresh")
        else:
            print(f"[Memory] No saved memory found, starting fresh")
        return {}

    def _save_memory(self):
        path = self._get_memory_path()
        try:
            with open(path, "wb") as f:
                pickle.dump(self.q_table, f)
        except Exception as e:
            print(f"[Memory] Save FAILED: {e}")