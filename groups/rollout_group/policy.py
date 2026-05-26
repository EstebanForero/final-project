import numpy as np
import random
from connect4.policy import Policy
from connect4.connect_state import ConnectState

class RolloutAgent(Policy):
    def __init__(self, num_simulations=40):
        self.num_simulations = num_simulations

    def mount(self, timeout=None):
        pass

    def _get_active_player(self, board):
        # Mapeo analítico del estado actual para identificar el turno
        return -1 if np.count_nonzero(board == -1) == np.count_nonzero(board == 1) else 1

    def act(self, s):
        yo = self._get_active_player(s)
        rival = -yo
        
        
        
        estado_actual = ConnectState(board=s, player=yo)
        columnas_libres = estado_actual.get_free_cols()
        
        if not columnas_libres:
            return 0


        try:
            tablero_espejo = s * -1
            estado_rival = ConnectState(board=tablero_espejo, player=yo)
            for col in columnas_libres:
                try:
                    if estado_rival.transition(col).get_winner() == yo:
                        return col
                except ValueError:
                    continue
        except:
            pass

        

        puntuacion_columnas = {}

        for col in columnas_libres:
            utilidad_total = 0.0
            
            for _ in range(self.num_simulations):
                try:
                    # Travectoria Online: Primer paso de la simulación
                    sim = estado_actual.transition(col)
                except ValueError:
                    break
                
                # Concepto: Default Policy / Muestreo Estocástico
                while not sim.is_final():
                    opciones = sim.get_free_cols()
                    if not opciones:
                        break
                    try:
                        sim = sim.transition(random.choice(opciones))
                    except ValueError:
                        break
                
            
                ganador = sim.get_winner()
                if ganador == yo:
                    utilidad_total += 1.0
                elif ganador == rival:
                    utilidad_total -= 1.0
            
            # Concepto: Estimación del valor Q(s,a) por promedio
            puntuacion_columnas[col] = utilidad_total / self.num_simulations

        if not puntuacion_columnas:
            return columnas_libres[0]

        # Concepto: Operador ArgMax / Selección de Acción
        return max(puntuacion_columnas, key=puntuacion_columnas.get)