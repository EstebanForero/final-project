# DEPTHAgent - Monte Carlo Tree Search para Connect-4

## Descripción

`DEPTHAgent` es un agente inteligente para jugar Connect-4 basado en **Monte Carlo Tree Search (MCTS)** con exploración multinivel del árbol de juego. A diferencia del agente básico UCB, este implementa un MCTS verdadero que:

- **Construye un árbol de juego** dinámicamente durante cada decisión
- **Explora múltiples profundidades** de nodos (no solo nivel 1)
- **Usa backpropagation** para actualizar todas las estadísticas del árbol
- **Aplica UCB1** para balancear explotación y exploración
- **Realiza playouts aleatorios** desde nodos expandidos hasta terminal

## Componentes

### 1. **MCTSNode**
Representa un nodo en el árbol de búsqueda con:
- Estado del tablero
- Estadísticas de visitas y recompensas
- Referencia a nodos hijos
- Acciones sin probar

### 2. **DEPTHAgent**
Agente que implementa los 4 pasos de MCTS:

#### Selection
Navega el árbol usando la fórmula UCB1:
```
UCB = Q(n) / N(n) + c * sqrt(ln(N(parent)) / N(n))
```

#### Expansion
Añade un nuevo nodo hijo cuando encuentra una acción sin probar

#### Simulation
Ejecuta un playout aleatorio desde el nodo expandido hasta terminal

#### Backpropagation
Actualiza las estadísticas de todos los nodos en el camino desde el expandido hasta la raíz

## Requisitos

```
numpy>=1.23
pydantic>=2.0
matplotlib>=3.0
```

Instalar con:
```bash
pip install numpy pydantic matplotlib
```

## Cómo Usar

### Opción 1: Ejecutar torneo completo

Desde la carpeta raíz del proyecto:
```bash
python main.py
```

El sistema automáticamente encontrará `DEPTHAgent` en esta carpeta y lo enfrentará contra otros agentes en un torneo.

### Opción 2: Usar el agente directamente

```python
from groups.Group_B_v3.policy import DEPTHAgent
from connect4.connect_state import ConnectState
import numpy as np

# Crear agente
agent = DEPTHAgent(num_simulations=500, max_depth=42)
agent.mount()

# Crear estado inicial
board = np.zeros((6, 7), dtype=int)
state = ConnectState(board=board, player=-1)

# Obtener acción
action = agent.act(board)
print(f"Agente elige columna: {action}")
```

## Parámetros

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `num_simulations` | 500 | Número de iteraciones MCTS por decisión |
| `max_depth` | 42 | Profundidad máxima del árbol (máximo movimientos posibles) |
| `c` | √2 | Constante de exploración UCB1 |

### Ajuste de Rendimiento

- **Aumentar `num_simulations`**: Mejor juego, pero más lento
- **Aumentar `max_depth`**: Explora más profundamente, pero costo computacional mayor
- **Reducir `c`**: Más explotación, menos exploración (y viceversa)

## Características Adicionales

El agente incluye heurísticas greedy:
1. **Detección de victoria inmediata**: Si puede ganar en 1 move, lo hace
2. **Bloqueo de derrota**: Si el oponente puede ganar, lo bloquea
3. **Fallback**: Si no hay acción MCTS, retorna primer movimiento legal


## Datos Necesarios

- **Tablero**: Array numpy (6, 7) con valores {-1 (rojo), 0 (vacío), 1 (amarillo)}
- **Jugador actual**: -1 (rojo) o 1 (amarillo)
- **Columnas libres**: Automáticamente detectadas del estado

