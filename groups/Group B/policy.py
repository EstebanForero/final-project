import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import override
import math
from collections import defaultdict


class UCBAgent(Policy):
  
    def __init__(self, num_simulations: int = 20, exploration_constant: float = 1.414):
        self.num_simulations = num_simulations
        self.c = exploration_constant
        self.player = None  # Will be determined by board state in act()

    @override
    def mount(self) -> None:
        """Initialize/setup the policy."""
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        # Determine current player based on piece count
        red_count = np.sum(s == -1)
        yellow_count = np.sum(s == 1)
        
        if red_count <= yellow_count:
            self.player = -1  # Red's turn
        else:
            self.player = 1   # Yellow's turn
        
        # Get current board state
        current_state = ConnectState(board=s, player=self.player)
        
        # Get legal actions
        legal_actions = get_legal_actions(s)
        
        if not legal_actions:
            return 0  # Fallback (shouldn't happen in valid game state)
        
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # Local action statistics: q[action] = {visits: int, value: float}
        action_stats = {a: {"visits": 0, "value": 0.0} for a in legal_actions}
        
        # Perform MCTS simulations
        for _ in range(self.num_simulations):
            # Selection + Expansion + Simulation + Backpropagation
            result = self._mcts_search(current_state, action_stats, legal_actions)
            # Update statistics for the action taken in root
            if result["action"] is not None:
                action_stats[result["action"]]["visits"] += 1
                action_stats[result["action"]]["value"] += result["value"]
        
        # UCB-based exploitation: select action with highest Q(s,a)
        best_action = max(
            legal_actions,
            key=lambda a: (
                action_stats[a]["value"] / action_stats[a]["visits"]
                if action_stats[a]["visits"] > 0
                else 0
            )
        )
        
        return int(best_action)

    def _mcts_search(
        self,
        state: ConnectState,
        action_stats: dict,
        legal_actions: list
    ) -> dict:
        # Selection: Choose action from root using UCB
        ucb_values = {}
        for action in legal_actions:
            visits = action_stats[action]["visits"]
            if visits == 0:
                # Unvisited actions get priority
                ucb_values[action] = float('inf')
            else:
                q_value = action_stats[action]["value"] / visits
                exploration = self.c * math.sqrt(math.log(sum(s["visits"] for s in action_stats.values()) + 1) / visits)
                ucb_values[action] = q_value + exploration
        
        selected_action = max(legal_actions, key=lambda a: ucb_values[a])
        
        # Apply selected action (Expansion of root)
        next_state = apply_action(state.board, selected_action, self.player)
        
        # Simulation: Random rollout from next state
        opponent_player = -self.player
        next_connect_state = ConnectState(board=next_state, player=opponent_player)
        
        outcome = random_rollout(next_connect_state)
        
        # Convert outcome to reward from our perspective (Red = -1)
        if outcome == 0:  # Draw
            reward = 0
        elif outcome == self.player:  # We won
            reward = 1.0
        else:  # Opponent won
            reward = -1.0
        
        return {"action": selected_action, "value": reward}

# Auxiliary Functions for MCTS-UCB
def get_legal_actions(board: np.ndarray) -> list[int]:
    legal_actions = []
    for col in range(7):
        if board[0, col] == 0:  # Column is not full
            legal_actions.append(col)
    return legal_actions


def apply_action(board: np.ndarray, action: int, player: int) -> np.ndarray:
    new_board = board.copy()
    
    # Find the lowest empty row in the column
    for row in reversed(range(6)):
        if new_board[row, action] == 0:
            new_board[row, action] = player
            break
    
    return new_board


def check_winner(board: np.ndarray, player: int) -> bool:
    rows, cols = 6, 7
    
    # Check horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if all(board[r, c + i] == player for i in range(4)):
                return True
    
    # Check vertical
    for c in range(cols):
        for r in range(rows - 3):
            if all(board[r + i, c] == player for i in range(4)):
                return True
    
    # Check diagonal (down-right)
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True
    
    # Check diagonal (down-left)
    for r in range(rows - 3):
        for c in range(3, cols):
            if all(board[r + i, c - i] == player for i in range(4)):
                return True
    
    return False


def is_terminal(board: np.ndarray) -> bool:
    # Check if board is full (draw)
    if not any(board[0] == 0):
        return True
    
    # Check if either player has won
    if check_winner(board, -1) or check_winner(board, 1):
        return True
    
    return False


def get_game_winner(board: np.ndarray) -> int:
    if check_winner(board, -1):
        return -1
    elif check_winner(board, 1):
        return 1
    else:
        return 0


def random_rollout(state: ConnectState) -> int:
    current_state = state
    rng = np.random.default_rng()
    
    while not is_terminal(current_state.board):
        # Get legal actions
        legal_actions = get_legal_actions(current_state.board)
        
        if not legal_actions:
            break
        
        # Choose random action
        action = rng.choice(legal_actions)
        
        # Apply action
        current_state = current_state.transition(action)
    
    # Return the winner
    return get_game_winner(current_state.board)


def mcts_search(board: np.ndarray, num_simulations: int = 20) -> int:
    policy = UCBAgent(num_simulations=num_simulations)
    return policy.act(board)
