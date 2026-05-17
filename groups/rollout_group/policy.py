import numpy as np
import random
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class RolloutAgent(Policy):
    def __init__(self, num_simulations=50):
        self.num_simulations = num_simulations

    def mount(self, timeout=None):
        # Evita el error de Gradescope aceptando cualquier argumento de tiempo
        pass

    def _get_active_player(self, board):
        # Cuenta fichas: si hay empate juega Rojo (-1), si no juega Amarillo (1)
        return -1 if np.count_nonzero(board == -1) == np.count_nonzero(board == 1) else 1

    def act(self, s):
        yo = self._get_active_player(s)
        rival = -yo
        
        # Crear el simulador local para planificar "Online"
        estado_actual = ConnectState(board=s, player=yo)
        columnas_libres = estado_actual.get_free_cols()
        
        if not columnas_libres:
            return 0

        # --- FILTRO 1: ¿Puedo ganar yo en este turno? ---
        for col in columnas_libres:
            if estado_actual.transition(col).get_winner() == yo:
                return col

        # --- FILTRO 2: ¿El rival ganará en su turno? (¡BLOQUEAR!) ---
        estado_rival = ConnectState(board=s, player=rival)
        for col in columnas_libres:
            if estado_rival.transition(col).get_winner() == rival:
                return col

        # --- MONTE CARLO ROLLOUTS (Si no hay peligro ni victoria inmediata) ---
        puntuacion_columnas = {}

        for col in columnas_libres:
            utilidad_total = 0.0
            
            for _ in range(self.num_simulations):
                # Avanzar un paso en la simulación mental
                sim = estado_actual.transition(col)
                
                # Política por Defecto: Terminar el juego al azar
                while not sim.is_final():
                    movimiento_azar = random.choice(sim.get_free_cols())
                    sim = sim.transition(movimiento_azar)
                
                # Evaluar el resultado según el Principio de Inversión de Signos
                if sim.get_winner() == yo:
                    utilidad_total += 1.0
                elif sim.get_winner() == rival:
                    utilidad_total -= 1.0
            
            # Guardar el valor Q estimado (Promedio de las simulaciones)
            puntuacion_columnas[col] = utilidad_total / self.num_simulations

        # Selección Codiciosa (Greedy): Elegir la columna con mayor Q-valor
        return max(puntuacion_columnas, key=puntuacion_columnas.get)