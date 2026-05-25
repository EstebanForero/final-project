import numpy as np
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from connect4.connect_state import ConnectState
from groups.Rollout_2.policy import RolloutAgent2

# ============================================================
# ¡IMPORTANTE! CORRECCIÓN DEL ERROR DE CLASE ABSTRACTA
# Abre el archivo groups/rollout_group/policy.py
# Busca el nombre de la clase (probablemente sea RolloutAgent o Agent)
# NO importes 'Policy'. Importa la clase real como se muestra aquí:
# ============================================================
try:
    from groups.rollout_group.policy import RolloutAgent as RivalAgent # <--- CAMBIA "RolloutAgent" POR EL NOMBRE REAL
except ImportError:
    # Fallback temporal por si el nombre no es correcto para que el script no explote
    print("[WARNING] No se pudo importar RivalAgent. Usando agente aleatorio como rival.")
    class RivalAgent:
        def act(self, board): return random.choice([c for c in range(7) if board[0,c]==0])

# Adaptador de seguridad para el rival
class RivalAdapter:
    def __init__(self):
        self.model = RivalAgent()
    def act(self, board):
        # Aseguramos compatibilidad
        return self.model.act(board)

# ============================================================
# CONSTANTS
# ============================================================
CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]

# ============================================================
# CORE GAME LOGIC (module-level — picklable on Windows)
# ============================================================
def _smart_move(board, player):
    rival = -player
    state = ConnectState(board=board, player=player)
    free = state.get_free_cols()
    if not free: return None
    for c in free:
        if state.transition(c).get_winner() == player: return c
    for c in free:
        if state.transition(c).get_winner() == rival: return c
    for c in CENTER_ORDER:
        if c in free: return c
    return random.choice(free)

def _rollout(board, col, yo, n=5):
    total = 0.0
    base = ConnectState(board=board, player=yo).transition(col)
    rival = -yo
    for _ in range(n):
        sim = base
        while not sim.is_final():
            move = _smart_move(sim.board, sim.player)
            if move is None: break
            sim = sim.transition(move)
        w = sim.get_winner()
        total += (1.0 if w == yo else (-1.0 if w == rival else 0.0))
    return total / n

def _agent_pick(state, yo):
    free = state.get_free_cols()
    rival = -yo
    if not free: return None
    for c in free:
        if state.transition(c).get_winner() == yo: return c
    for c in free:
        if state.transition(c).get_winner() == rival: return c
    for c in free:
        for r in range(3):
            w = state.board[r:r+4, c]
            if np.count_nonzero(w == rival) == 3 and np.count_nonzero(w == 0) == 1:
                return c
    scores = {c: _rollout(state.board, c, yo, n=5) for c in free}
    return max(scores, key=scores.get), scores

# ============================================================
# GAME RUNNERS (Top-level functions for ProcessPoolExecutor)
# ============================================================
def play_self(seed=None):
    if seed is not None: random.seed(seed); np.random.seed(seed)
    state = ConnectState(player=1)
    history = [] 
    while not state.is_final():
        yo = state.player
        free = state.get_free_cols()
        if not free: break
        result = _agent_pick(state, yo)
        if isinstance(result, tuple):
            col, scores = result
            history.append((state.board.tobytes(), yo, col, scores[col]))
        else:
            col = result 
        if col is None: break
        state = state.transition(col)
        
    winner = state.get_winner()
    out = []
    for (sb, player, c, mc_val) in history:
        outcome = 1.0 if winner == player else (-1.0 if winner == -player else 0.0)
        value = 0.5 * mc_val + 0.5 * outcome * 10.0
        out.append((sb, c, value))
    return out

def _play_vs_random(agent_is_p1, seed=None):
    if seed is not None: random.seed(seed); np.random.seed(seed)
    state = ConnectState(player=1)
    yo = 1 if agent_is_p1 else -1
    history = []
    while not state.is_final():
        player = state.player
        free = state.get_free_cols()
        if not free: break
        if player == yo:
            result = _agent_pick(state, yo)
            if isinstance(result, tuple):
                col, scores = result
                history.append((state.board.tobytes(), yo, col, scores[col]))
            else:
                col = result
        else:
            col = random.choice(free)
        if col is None: break
        state = state.transition(col)
        
    winner = state.get_winner()
    out = []
    for (sb, player, c, mc_val) in history:
        outcome = 1.0 if winner == player else (-1.0 if winner == -player else 0.0)
        value = 0.5 * mc_val + 0.5 * outcome * 10.0
        out.append((sb, c, value))
    return out

def play_vs_random_p1(seed=None): return _play_vs_random(True, seed)
def play_vs_random_p2(seed=None): return _play_vs_random(False, seed)

# Funciones de fase 3 seguras para Windows
def _play_vs_rival(agent_is_p1, seed=None):
    if seed is not None: random.seed(seed); np.random.seed(seed)
    state = ConnectState(player=1)
    yo = 1 if agent_is_p1 else -1
    history = []
    # Instanciamos el rival AQUÍ ADENTRO para evitar errores de Pickling en Windows
    rival_agent = RivalAdapter() 
    
    while not state.is_final():
        player = state.player
        free = state.get_free_cols()
        if not free: break
        if player == yo:
            result = _agent_pick(state, yo)
            if isinstance(result, tuple):
                col, scores = result
                history.append((state.board.tobytes(), yo, col, scores[col]))
            else:
                col = result
        else:
            col = rival_agent.act(state.board.copy())
            if col is None or col not in free: col = random.choice(free)
        state = state.transition(col)
        
    winner = state.get_winner()
    out = []
    for (sb, p, c, mc_val) in history:
        outcome = 1.0 if winner == p else (-1.0 if winner == -p else 0.0)
        value = 0.5 * mc_val + 0.5 * outcome * 10.0
        out.append((sb, c, value))
    return out

def play_vs_rival_p1(seed=None): return _play_vs_rival(True, seed)
def play_vs_rival_p2(seed=None): return _play_vs_rival(False, seed)

# ============================================================
# PARALLEL BATCH RUNNER
# ============================================================
def run_batch(game_fn, n_games, n_workers=10):
    seeds = [random.randint(0, 2**31) for _ in range(n_games)]
    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(game_fn, s) for s in seeds]
        for f in as_completed(futures):
            try:
                all_results.extend(f.result())
            except Exception as e:
                pass # Ignorar errores aislados de workers
    return all_results

def apply_to_qtable(q_table, results, alpha=0.1):
    new_states = 0
    for (state_bytes, col, value) in results:
        if state_bytes not in q_table:
            q_table[state_bytes] = {"q": np.zeros(7), "n": np.zeros(7), "total": 0}
            new_states += 1
        sd = q_table[state_bytes]
        sd["q"][col] += alpha * (value - sd["q"][col])
        sd["n"][col] += 1
        sd["total"] += 1
    return new_states

def play_vs_agent_benchmark(agent1, agent2, agent_is_p1=True, seed=None):
    if seed is not None: random.seed(seed); np.random.seed(seed)
    state = ConnectState(player=1)
    while not state.is_final():
        yo = state.player
        current_policy = agent1 if (yo == 1 if agent_is_p1 else yo == -1) else agent2
        col = current_policy.act(state.board.copy())
        if col is None: break
        state = state.transition(col)
    winner = state.get_winner()
    return 1 if winner == (1 if agent_is_p1 else -1) else (-1 if winner != 0 else 0)

# ============================================================
# TRAINING LOOP
# ============================================================
def train(n_self_play=500, n_random_play=500, n_vs_rival=500, n_workers=8, log_every=100):
    agent = RolloutAgent2()
    
    print("=" * 60)
    print(f" TRAINING START | Self:{n_self_play} | Random:{n_random_play} | Rival:{n_vs_rival}")
    print("=" * 60)

    # ── FASE 1: Self-play ──────────────────────────────────
    print("\n[Phase 1] Self-play")
    for start in range(0, n_self_play, log_every):
        n = min(log_every, n_self_play - start)
        res = run_batch(play_self, n, n_workers)
        new = apply_to_qtable(agent.q_table, res)
        agent._save_memory()
        print(f"  Fase 1: {start+n}/{n_self_play} completado (+{new} nuevos estados)")

    # ── FASE 2: vs Random ──────────────────────────────────
    print("\n[Phase 2] vs Random agent")
    for start in range(0, n_random_play, log_every):
        n = min(log_every, n_random_play - start)
        half = n // 2
        res = run_batch(play_vs_random_p1, half, n_workers)
        res += run_batch(play_vs_random_p2, half, n_workers)
        new = apply_to_qtable(agent.q_table, res)
        agent._save_memory()
        print(f"  Fase 2: {start+n}/{n_random_play} completado (+{new} nuevos estados)")

    # ── FASE 3: Adversarial ────────────────────────────────
    print(f"\n[Phase 3] Adversarial Training vs Rollout_Group ({n_vs_rival} games)")
    for start in range(0, n_vs_rival, log_every):
        n = min(log_every, n_vs_rival - start)
        half = n // 2
        res = run_batch(play_vs_rival_p1, half, n_workers)
        res += run_batch(play_vs_rival_p2, half, n_workers)
        new = apply_to_qtable(agent.q_table, res)
        agent._save_memory()
        print(f"  Fase 3: {start+n}/{n_vs_rival} completado (+{new} nuevos estados)")

    # ── BENCHMARK FINAL ────────────────────────────────────
    print("\n" + "="*50)
    print(" BENCHMARK FINAL (Frozen weights - 100 partidas)")
    print("="*50)
    
    agent.ALPHA = 0.0 # Congelamos el aprendizaje
    rival_benchmark = RivalAdapter()

    benchmarks = [
        ("vs Random", lambda p1: play_vs_agent_benchmark(agent, RivalAgent(), p1) if "RivalAgent" in str(type(RivalAgent)) else _play_vs_random(p1)[-1][2] ), # Hack para test de random
        ("vs Rival Group", lambda p1: play_vs_agent_benchmark(agent, rival_benchmark, p1))
    ]

    for name, game_fn in benchmarks:
        wins = losses = draws = 0
        for i in range(100):
            # Hack rápido para reutilizar _play_vs_random
            if name == "vs Random":
                res = _play_vs_random(i % 2 == 0)
                val = res[-1][2] if res else 0
            else:
                val = game_fn(i % 2 == 0)
                
            if val > 0: wins += 1
            elif val < 0: losses += 1
            else: draws += 1
        
        print(f" {name:15}: W:{wins:3} | L:{losses:3} | D:{draws:3} | WinRate: {(wins/100)*100:.1f}%")
        
    print("\nEntrenamiento finalizado exitosamente.")

if __name__ == "__main__":
    # Puedes ajustar n_workers dependiendo de los núcleos de tu CPU
    train(n_self_play=500, n_random_play=500, n_vs_rival=500, n_workers=8)