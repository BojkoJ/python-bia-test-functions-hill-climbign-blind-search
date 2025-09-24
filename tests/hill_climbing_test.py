import sys
import os

# Nastavení cesty k projektu pro import main.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import hill_climbing, sphere, get_default_bounds

if __name__ == "__main__":
    bounds = get_default_bounds("sphere", 2)
    # Spuštění hill climbing
    hill_climbing(
        sphere,
        bounds,
        max_iter=150,
        step_sigma=0.18,
        neighbours=25,
        seed=42,
        early_stop_no_improve=40,
        visualize=True,
        pause_seconds=0.5,
        num_points=70,
        surface_alpha=0.35,  # poloprůhlednost povrchu pro lepší viditelnost
    )