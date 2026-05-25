
import numpy as np
import random
from connect4.policy import Policy
from connect4.connect_state import ConnectState
import math


class MCTSNode:
    """Node in the MCTS tree."""
    
    def __init__(self, state: ConnectState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children = {}  # Map from action to MCTSNode
        self.visits = 0
        self.value = 0.0  # Sum of rewards
        self.untried_actions = None
    
    def get_untried_actions(self):
        """Get actions that haven't been tried yet from this node."""
        if self.untried_actions is None:
            self.untried_actions = list(self.state.get_free_cols())
        return self.untried_actions
    
    def is_fully_expanded(self):
        """Check if all children have been visited at least once."""
        return len(self.get_untried_actions()) == 0
    
    def best_child(self, c: float):
        """Select child with highest UCB value."""
        if not self.children:
            return None
        
        best = None
        best_ucb = -float('inf')
        
        for action, child in self.children.items():
            if child.visits == 0:
                ucb = float('inf')
            else:
                q_value = child.value / child.visits
                exploration = c * math.sqrt(math.log(self.visits) / child.visits)
                ucb = q_value + exploration
            
            if ucb > best_ucb:
                best_ucb = ucb
                best = (action, child)
        
        return best
    
    def add_child(self, action: int):
        """Add a child node for the given action."""
        try:
            next_state = self.state.transition(action)
            child = MCTSNode(state=next_state, parent=self, action=action)
            self.children[action] = child
            self.untried_actions.remove(action)
            return child
        except ValueError:
            return None
    
    def update(self, reward: float):
        """Update this node's statistics."""
        self.visits += 1
        self.value += reward


class DEPTHAgent(Policy):
    """True Monte Carlo Tree Search Agent for Connect-4 with deep tree exploration."""

    def __init__(self, num_simulations: int = 500, max_depth: int = 42):
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c = math.sqrt(2)  # UCB exploration constant

    def mount(self, timeout=None):   
        pass

    def act(self, s):
        try:
            yo = self._get_active_player(s)
            current_state = ConnectState(board=s, player=yo)
            
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
                    temp_board = s.copy()
                    placed = False
                    for row in reversed(range(6)):
                        if temp_board[row, col] == 0:
                            temp_board[row, col] = opponent
                            placed = True
                            break
                    
                    if not placed:
                        continue
                    
                    opponent_state = ConnectState(board=temp_board, player=opponent)
                    if opponent_state.get_winner() == opponent:
                        return int(col)
                except Exception:
                    continue
            
            # True MCTS tree search
            root = MCTSNode(state=current_state)
            
            for _ in range(self.num_simulations):
                self._mcts_iteration(root, yo)
            
            # Select best action from root children
            best_action = None
            best_visits = -1
            
            for action, child in root.children.items():
                if child.visits > best_visits:
                    best_visits = child.visits
                    best_action = action
            
            if best_action is not None:
                return int(best_action)
            
            # Fallback
            return int(free_cols[0])
        
        except Exception as e:
            try:
                current_state = ConnectState(board=s, player=self._get_active_player(s))
                free_cols = current_state.get_free_cols()
                if free_cols:
                    return int(free_cols[0])
                return 0
            except:
                return 0

    def _mcts_iteration(self, root: MCTSNode, player: int):
        """Single MCTS iteration: selection -> expansion -> simulation -> backpropagation."""
        
        # Selection phase: traverse tree using UCB
        node = root
        depth = 0
        
        while node.is_fully_expanded() and node.children and depth < self.max_depth:
            best_result = node.best_child(self.c)
            if best_result is None:
                break
            action, node = best_result
            depth += 1
        
        # Expansion phase: add a new child if not terminal
        if not node.state.is_final() and not node.is_fully_expanded() and depth < self.max_depth:
            untried = node.get_untried_actions()
            if untried:
                action = random.choice(untried)
                child = node.add_child(action)
                if child is not None:
                    node = child
                    depth += 1
        
        # Simulation phase: random playout from selected node
        reward = self._random_playout(node.state, player, depth)
        
        # Backpropagation phase: update all nodes in path
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def _random_playout(self, state: ConnectState, player: int, current_depth: int = 0) -> float:
        """Random playout from state to terminal, return reward from player's perspective."""
        current = state
        depth = current_depth
        
        while not current.is_final() and depth < self.max_depth:
            free_cols = current.get_free_cols()
            if not free_cols:
                break
            
            col = random.choice(free_cols)
            try:
                current = current.transition(col)
                depth += 1
            except ValueError:
                break
        
        winner = current.get_winner()
        
        # Return reward from player's perspective
        if winner == 0:
            return 0.0
        elif winner == player:
            return 1.0
        else:
            return -1.0

    def _get_active_player(self, board):
        """Determine whose turn it is."""
        ones = np.count_nonzero(board == 1)
        neg_ones = np.count_nonzero(board == -1)
        return -1 if ones == neg_ones else 1
