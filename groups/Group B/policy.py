
import numpy as np
import random
from connect4.policy import Policy
from connect4.connect_state import ConnectState
import math


class UCBAgent(Policy):
    """Online Policy Improvement with local MCTS-UCB exploration for Connect-4."""

    def __init__(self, num_simulations: int = 500):
        self.num_simulations = num_simulations
        self.c = math.sqrt(2)  # UCB exploration constant

    def mount(self, timeout=None):   
        pass

    def act(self, s):
        try:
            # Determine current player
            yo = self._get_active_player(s)
            current_state = ConnectState(board=s, player=yo)
            
            # Get legal actions
            free_cols = current_state.get_free_cols()
            
            if not free_cols:
                return int(list(range(7))[0])
            
            if len(free_cols) == 1:
                return int(free_cols[0])
            
            # Greedy win check - immediate victory
            for col in free_cols:
                try:
                    next_state = current_state.transition(col)
                    if next_state.get_winner() == yo:
                        return int(col)
                except ValueError:
                    continue
            
            # Greedy block check - prevent immediate opponent loss
            opponent = -yo
            for col in free_cols:
                try:
                    # Create a state where opponent plays this column
                    temp_board = s.copy()
                    # Find where piece would land
                    placed = False
                    for row in reversed(range(6)):
                        if temp_board[row, col] == 0:
                            temp_board[row, col] = opponent
                            placed = True
                            break
                    
                    if not placed:
                        continue
                    
                    # Check if opponent wins
                    opponent_state = ConnectState(board=temp_board, player=opponent)
                    if opponent_state.get_winner() == opponent:
                        return int(col)  # Block this column
                except Exception:
                    continue
            
            # MCTS-UCB search for best action
            action_stats = {col: {"visits": 0, "value": 0.0} for col in free_cols}
            
            for _ in range(self.num_simulations):
                result = self._mcts_trial(current_state, action_stats, free_cols, yo)
                if result["action"] is not None:
                    action_stats[result["action"]]["visits"] += 1
                    action_stats[result["action"]]["value"] += result["value"]
            
            # Select best action by average reward
            best_action = max(
                free_cols,
                key=lambda col: (
                    action_stats[col]["value"] / action_stats[col]["visits"]
                    if action_stats[col]["visits"] > 0
                    else 0.0
                )
            )
            
            return int(best_action)
        
        except Exception as e:
            # Fallback: return first legal action
            try:
                current_state = ConnectState(board=s, player=self._get_active_player(s))
                free_cols = current_state.get_free_cols()
                if free_cols:
                    return int(free_cols[0])
                return 0
            except:
                return 0

    def _mcts_trial(self, state: ConnectState, action_stats: dict, free_cols: list, player: int) -> dict:
        """Single MCTS trial with UCB selection."""
        # UCB selection
        total_visits = sum(s["visits"] for s in action_stats.values()) + 1
        
        ucb_values = {}
        for col in free_cols:
            visits = action_stats[col]["visits"]
            if visits == 0:
                ucb_values[col] = float('inf')
            else:
                q_value = action_stats[col]["value"] / visits
                exploration = self.c * math.sqrt(math.log(total_visits) / visits)
                ucb_values[col] = q_value + exploration
        
        selected_col = int(max(free_cols, key=lambda col: ucb_values[col]))
        
        # Simulate from next state
        try:
            next_state = state.transition(selected_col)
        except ValueError:
            return {"action": selected_col, "value": 0.0}
        
        opponent = -player
        outcome = self._random_playout(next_state)
        
        # Reward from player's perspective
        if outcome == 0:
            reward = 0.0
        elif outcome == player:
            reward = 1.0
        else:
            reward = -1.0
        
        return {"action": selected_col, "value": reward}

    def _random_playout(self, state: ConnectState) -> int:
        """Random playout from state to terminal."""
        current = state
        
        while not current.is_final():
            free_cols = current.get_free_cols()
            if not free_cols:
                break
            
            col = random.choice(free_cols)
            try:
                current = current.transition(col)
            except ValueError:
                break
        
        return current.get_winner()

    def _get_active_player(self, board):
        """Determine whose turn it is."""
        ones = np.count_nonzero(board == 1)
        neg_ones = np.count_nonzero(board == -1)
        return -1 if ones == neg_ones else 1
