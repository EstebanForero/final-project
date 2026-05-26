# RolloutAgent2 — Guía de Uso

Agente de Connect4 basado en Monte Carlo Rollouts con Q-learning tabular, heurísticas de decisión y memoria persistente.

---

## 1. Instalación y Dependencias

Instala todas las dependencias necesarias (incluyendo las del agente y las del notebook de análisis) ejecutando:

```bash
pip install -r requirements.txt
```
## 2. Cómo correr el Torneo

Para ejecutar el torneo automático entre todos los agentes disponibles en la carpeta groups/, usa:

```bash
python main.py
```

3. Entrenamiento previo (Recomendado)

Para acumular conocimiento en cerebro.bin antes de competir en el torneo, ejecuta el script de entrenamiento en paralelo:
```Bash

python train.py
```

Configuración del Entrenamiento

Puedes modificar las siguientes variables directamente al inicio de train.py:

    N_GAMES_TOTAL (Por defecto: 2000): Cantidad total de partidas a simular.

    SIMS_TRAINING (Por defecto: 10): Simulaciones internas de rollout por cada turno.

    N_WORKERS: Número de núcleos de CPU en paralelo (usa el máximo disponible por defecto).

