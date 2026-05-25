import numpy as np
import random
import pickle
import os
from connect4.policy import Policy
from connect4.connect_state import ConnectState


# ============================================================
# MÓDULO 1: HEURISTICS — Reward Shaping
# ============================================================
# [HEURISTIC] This entire class implements reward shaping —
# a heuristic technique that adds domain knowledge to the reward
# signal instead of relying only on win/loss outcomes.
# Rather than waiting until the game ends, we score intermediate
# board states based on known Connect 4 patterns (3-in-a-row,
# 2-in-a-row, center control). This guides the Monte Carlo
# rollouts toward strategically better terminal states.
class HeuristicEvaluator:
    # [HEURISTIC] Column weights encode the geometric fact that
    # center columns participate in more winning lines than edges.
    # Col 3 (center) = 0.20, edges = 0.02.
    COL_WEIGHT = np.array([0.02, 0.06, 0.12, 0.20, 0.12, 0.06, 0.02])

    @staticmethod
    def get_shaping_reward(board, yo):
        rival, score = -yo, 0.0

        def scan(window):
            nonlocal score
            m = np.count_nonzero(window==yo)
            e = np.count_nonzero(window==rival)
            v = np.count_nonzero(window==0)
            if m>0 and e>0: return  # mixed window, no value

            # [HEURISTIC] Score windows by pattern strength.
            # 3-in-a-row with 1 empty = near-win threat (+0.6 / -0.7).
            # 2-in-a-row with 2 empty = building threat (+0.2 / -0.3).
            # Opponent threats weighted slightly higher to prioritise defence.
            if m==3 and v==1: score += 0.6
            if m==2 and v==2: score += 0.2
            if e==3 and v==1: score -= 0.7
            if e==2 and v==2: score -= 0.3

        # [HEURISTIC] Scan all 4 directions for threat windows
        for r in range(6):
            for c in range(4):
                scan(board[r, c:c+4])          # horizontal
        for c in range(7):
            for r in range(3):
                scan(board[r:r+4, c])          # vertical
        for r in range(3):
            for c in range(4):
                scan(np.array([board[r+i,   c+i] for i in range(4)]))  # diagonal ↘
                scan(np.array([board[r+3-i, c+i] for i in range(4)]))  # diagonal ↗

        # [HEURISTIC] Center ownership bonus — reward occupying
        # high-value columns regardless of threat patterns.
        for c in range(7):
            score += np.count_nonzero(board[:,c]==yo) * HeuristicEvaluator.COL_WEIGHT[c]

        return score


# ============================================================
# MÓDULO 2: ROLLOUT AGENT
# ============================================================
class RolloutAgent2(Policy):
    N_ROLLOUTS   = 40     # simulations per action for MC evaluation
    ALPHA        = 0.1    # Q-learning rate — how fast Q-values update
    CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]

    # [HEURISTIC] Static center bias added to rollout scores before argmax.
    # Breaks ties consistently toward center columns without overriding
    # strong MC signals. Same shape as COL_WEIGHT.
    CENTER_BIAS  = np.array([0.02, 0.06, 0.12, 0.20, 0.12, 0.06, 0.02])

    def __init__(self):
        # [Q-TABLE] Load persisted Q-values from disk. The Q-table maps
        # board state → per-column Q-values learned across all past games.
        self.q_table = self._load_memory()

    def mount(self, timeout=None):
        # Called by the tournament before every game.
        # Save whatever the previous instance learned, then reload
        # so Q-values accumulate across games even though the
        # tournament creates a fresh object each time.
        self._save_memory()  # [Q-TABLE] flush previous game's learning
        self.q_table = self._load_memory()  # [Q-TABLE] reload latest values

    # ----------------------------------------------------------
    # PLAYER DETECTION
    # ----------------------------------------------------------
    @staticmethod
    def _get_player(board):
        # Player 1 always goes first, so equal piece counts = player 1's turn
        return 1 if np.count_nonzero(board==1)==np.count_nonzero(board==-1) else -1

    # ----------------------------------------------------------
    # EXPLICIT WIN CHECK
    # ----------------------------------------------------------
    @staticmethod
    def _can_win(board, col, player):
        # [HEURISTIC] Depth-1 greedy search — simulate dropping a piece
        # and check all 4 directions for an immediate win.
        # Used in Filter 1 (take win) and Filter 2 (block opponent win).
        if board[0, col] != 0:
            return False
        b = board.copy()
        for r in range(5, -1, -1):
            if b[r, col] == 0:
                b[r, col] = player
                break
        for r in range(6):
            for c in range(7):
                p = b[r,c]
                if p != player: continue
                if c+3<7 and all(b[r,c+i]==p for i in range(4)): return True
                if r+3<6 and all(b[r+i,c]==p for i in range(4)): return True
                if r+3<6 and c+3<7 and all(b[r+i,c+i]==p for i in range(4)): return True
                if r+3<6 and c-3>=0 and all(b[r+i,c-i]==p for i in range(4)): return True
        return False

    # ----------------------------------------------------------
    # THREAT DETECTORS
    # ----------------------------------------------------------
    def _detect_vertical_threat(self, board, col, player):
        # [HEURISTIC] Detect 3-in-a-column with 1 empty above —
        # catches the edge-column stacking exploit seen in lost games.
        for r in range(3):
            w = board[r:r+4, col]
            if np.count_nonzero(w==player)==3 and np.count_nonzero(w==0)==1:
                return True
        return False

    def _detect_diagonal_threat(self, board, col, rival):
        # [HEURISTIC] Simulate placing rival's piece and check if it
        # creates or extends a diagonal 3-in-a-row. Prevents diagonal
        # build-ups that pure rollouts sometimes miss.
        test = board.copy()
        row  = None
        for r in range(5,-1,-1):
            if test[r,col]==0: row=r; break
        if row is None: return False
        test[row,col] = rival
        for r in range(3):
            for c in range(4):
                for w in (
                    np.array([test[r+i,   c+i] for i in range(4)]),
                    np.array([test[r+3-i, c+i] for i in range(4)])
                ):
                    if np.count_nonzero(w==rival)==3 and np.count_nonzero(w==0)==1:
                        return True
        return False

    def _count_threats(self, board, col, player):
        # [HEURISTIC] Count simultaneous 3-in-a-row threats after playing col.
        # If >= 2, opponent can only block one → guaranteed win next turn.
        # Used in Filter 3 (create fork) and Filter 4 (block opponent fork).
        b = board.copy()
        for r in range(5,-1,-1):
            if b[r,col]==0: b[r,col]=player; break
        rival   = -player
        threats = 0
        def scan_threats(window):
            nonlocal threats
            m = np.count_nonzero(window==player)
            e = np.count_nonzero(window==rival)
            v = np.count_nonzero(window==0)
            if m==3 and v==1 and e==0: threats+=1
        for r in range(6):
            for c in range(4): scan_threats(b[r,c:c+4])
        for c in range(7):
            for r in range(3): scan_threats(b[r:r+4,c])
        for r in range(3):
            for c in range(4):
                scan_threats(np.array([b[r+i,   c+i] for i in range(4)]))
                scan_threats(np.array([b[r+3-i, c+i] for i in range(4)]))
        return threats

    # ----------------------------------------------------------
    # MAIN ENTRY POINT
    # ----------------------------------------------------------
    def act(self, s):
        yo    = self._get_player(s)
        rival = -yo
        free  = [c for c in range(7) if s[0,c]==0]

        if not free:
            return 0

        # [HEURISTIC] FILTER 1 — depth-1 greedy win search.
        # Always take an immediate win before any other consideration.
        for col in free:
            if self._can_win(s, col, yo):
                return col

        # [HEURISTIC] FILTER 2 — depth-1 greedy block.
        # Block any immediate opponent win.
        for col in free:
            if self._can_win(s, col, rival):
                return col

        # [HEURISTIC] FILTER 3 — fork creation.
        # If a move creates 2+ simultaneous threats, take it — opponent
        # can only block one, so this guarantees a win next turn.
        for col in free:
            if self._count_threats(s, col, yo) >= 2:
                return col

        # [HEURISTIC] FILTER 4 — fork prevention.
        # Block the opponent from creating an unblockable double threat.
        for col in free:
            if self._count_threats(s, col, rival) >= 2:
                return col

        # [HEURISTIC] FILTER 5 — vertical build-up block.
        # Catches 3-in-a-column before it becomes an unstoppable stack.
        for col in free:
            if self._detect_vertical_threat(s, col, rival):
                return col

        # [HEURISTIC] FILTER 6 — diagonal build-up block.
        # Catches diagonal 3-in-a-rows before they complete.
        for col in free:
            if self._detect_diagonal_threat(s, col, rival):
                return col

        # ── No forcing move found — fall through to MC + Q-table ──

        # [Q-TABLE] Look up (or initialise) the Q-value entry for this
        # exact board state. Each entry stores:
        #   "q"     — learned Q-value per column (7 floats)
        #   "n"     — visit count per column (for tracking)
        #   "total" — total visits to this state
        state_id = s.tobytes()
        if state_id not in self.q_table:
            self.q_table[state_id] = {"q":np.zeros(7),"n":np.zeros(7),"total":0}
            self._save_memory()  # persist new state immediately

        sd     = self.q_table[state_id]
        scores = {}

        for col in free:
            # [ONLINE POLICY IMPROVEMENT] Run Monte Carlo rollouts to
            # estimate Q(s, col) — the expected return from playing col
            # in state s. This is decision-time planning: we improve the
            # policy right now, during the actual game, rather than
            # only between games.
            val = self._monte_carlo_evaluation(s, col, yo)

            # [Q-TABLE] Blend MC estimate with stored Q-value.
            # q_bon adds a small weight (0.1×) from past experience so
            # states we've seen many times get a slight nudge toward
            # historically good moves without overriding fresh MC signal.
            q_bon = sd["q"][col] * 0.1

            # [HEURISTIC] Center bias nudges toward structurally better
            # columns when rollout scores are close. Col 3 = +0.20 bonus.
            c_bon = self.CENTER_BIAS[col]

            scores[col] = val + q_bon + c_bon

            # [Q-TABLE / ONLINE POLICY IMPROVEMENT] TD update toward the
            # MC estimate. This is the Q-learning update rule:
            #   Q(s,a) ← Q(s,a) + α * (target - Q(s,a))
            # where target = MC return for this action.
            # Executed on every turn — the policy improves online as
            # each new MC estimate refines the stored Q-value.
            sd["q"][col] += self.ALPHA * (val - sd["q"][col])  # [Q-TABLE WRITE]
            sd["n"][col] += 1
            sd["total"]  += 1

        # [ONLINE POLICY IMPROVEMENT] ArgMax over blended scores —
        # the improved policy selects the action with highest combined
        # MC estimate + Q-memory + center bias.
        return max(scores, key=scores.get)

    # ----------------------------------------------------------
    # MONTE CARLO EVALUATION
    # ----------------------------------------------------------
    def _monte_carlo_evaluation(self, s, col, yo):
        # [ONLINE POLICY IMPROVEMENT] Pure rollout evaluation:
        # simulate N_ROLLOUTS complete games from state s after
        # playing col, then average the outcomes. This is flat Monte
        # Carlo — no tree, no branching — giving an unbiased estimate
        # of Q(s, col) purely through sampling.
        total = 0.0
        rival = -yo
        base  = ConnectState(board=s, player=yo).transition(col)

        for _ in range(self.N_ROLLOUTS):
            sim = base
            while not sim.is_final():
                sim = sim.transition(self._rollout_policy(sim))
            w = sim.get_winner()
            # Terminal reward: win=+10, loss=-10, draw=0
            r = 10.0 if w==yo else (-10.0 if w==rival else 0.0)
            # [HEURISTIC] Add shaped reward to terminal state score —
            # this augments the sparse win/loss signal with board pattern
            # information, giving the MC estimate more gradient to work with.
            total += r + HeuristicEvaluator.get_shaping_reward(sim.board, yo)

        return total / self.N_ROLLOUTS

    def _rollout_policy(self, sim):
        # [HEURISTIC] Rollout default policy — not pure random.
        # Checks for immediate wins first (greedy depth-1), then
        # uses center-biased sampling. Smarter rollouts produce
        # better MC estimates than uniform random.
        free   = sim.get_free_cols()
        player = sim.player

        # [HEURISTIC] Win check inside rollout — grab forced wins
        for c in free:
            if sim.transition(c).get_winner() == player:
                return c

        # [HEURISTIC] Center-biased random — 45% chance of preferring
        # center columns. Makes simulated games more realistic.
        for c in self.CENTER_ORDER:
            if c in free and random.random() < 0.45:
                return c
        return random.choice(free)

    # ----------------------------------------------------------
    # PERSISTENCE — Q-table saved to disk as cerebro.bin
    # ----------------------------------------------------------
    def _get_memory_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cerebro.bin")

    def _load_memory(self):
        # [Q-TABLE] Deserialise Q-table from disk. Enables learning
        # to persist across tournament rounds and training sessions.
        path = self._get_memory_path()
        if os.path.exists(path):
            try:
                with open(path,"rb") as f:
                    data = pickle.load(f)
                print(f"[Memory] Loaded {len(data)} states from {path}")
                return data
            except Exception as e:
                print(f"[Memory] Load failed ({e}), starting fresh")
        else:
            print(f"[Memory] No file at {path}, starting fresh")
        return {}

    def _save_memory(self):
        # [Q-TABLE] Serialise Q-table to disk so learned values survive
        # between games. Called on new state discovery and at mount().
        path = self._get_memory_path()
        try:
            with open(path,"wb") as f:
                pickle.dump(self.q_table, f)
            print(f"[Memory] Saved {len(self.q_table)} states to {path}")
        except Exception as e:
            print(f"[Memory] Save FAILED: {e}")