import numpy as np
import random
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class RolloutAgent(Policy):
    def __init__(self, num_simulations: int = 50):
        self.num_simulations = num_simulations

    def mount(self) -> None:
        pass

    def _determine_active_player(self, board: np.ndarray) -> int:
        num_red = np.count_nonzero(board == -1)
        num_yellow = np.count_nonzero(board == 1)
        return -1 if num_red == num_yellow else 1

    def act(self, s: np.ndarray) -> int:
        my_player_marker = self._determine_active_player(s)
        current_state = ConnectState(board=s, player=my_player_marker)
        legal_actions = current_state.get_free_cols()
        
        if not legal_actions:
            return 0
            
        action_scores = {}
        
        for action in legal_actions:
            accumulated_utility = 0.0
            
            for _ in range(self.num_simulations):
                sim_state = current_state.transition(action)
                
                while not sim_state.is_final():
                    possible_moves = sim_state.get_free_cols()
                    random_move = random.choice(possible_moves)
                    sim_state = sim_state.transition(random_move)
                
                winner = sim_state.get_winner()
                if winner == my_player_marker:
                    accumulated_utility += 1.0
                elif winner == -my_player_marker:
                    accumulated_utility -= 1.0
                else:
                    accumulated_utility += 0.0
            
            action_scores[action] = accumulated_utility / self.num_simulations

        best_action = max(action_scores, key=action_scores.get)
        return best_action