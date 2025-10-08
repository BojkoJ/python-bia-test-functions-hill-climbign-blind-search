"""
Test pro Simulated Annealing s vizualizací na Ackley funkci.
"""
import sys
import os

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('TkAgg')  # Explicitní backend pro zobrazení oken

from main import simulated_annealing, ackley, get_default_bounds

def test_sa_ackley_with_visualization():
    """
    Spustí Simulated Annealing na Ackley funkci s heatmap vizualizací.
    """
    print("=" * 60)
    print("SIMULATED ANNEALING - Ackley funkce (2D)")
    print("=" * 60)
    
    bounds = get_default_bounds("ackley", dim=2)
    
    # Použijeme užší bounds pro lepší vizualizaci (jako na obrázku)
    bounds_viz = [(-10.0, 10.0), (-10.0, 10.0)]
    
    best_x, best_f = simulated_annealing(
        objective=ackley,
        bounds=bounds_viz,
        max_iter=1000,
        T_initial=100.0,
        T_min=0.001,
        alpha=0.95,
        step_sigma=1.0,
        seed=42,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (0, 0), f = 0")
    print()


def test_sa_sphere():
    """
    Rychlý test SA na Sphere bez vizualizace.
    """
    from main import sphere
    
    print("=" * 60)
    print("SIMULATED ANNEALING - Sphere funkce (test)")
    print("=" * 60)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    
    best_x, best_f = simulated_annealing(
        objective=sphere,
        bounds=bounds,
        max_iter=500,
        T_initial=50.0,
        T_min=0.01,
        alpha=0.95,
        step_sigma=0.5,
        seed=123,
        visualize=False,
    )
    
    print(f"Výsledek: x = {best_x}, f(x) = {best_f:.8f}")
    
    # Kontrola: pro Sphere by mělo být blízko nuly
    assert best_f < 0.1, f"SA na Sphere selhalo: f = {best_f} > 0.1"
    print("✓ Test prošel!")
    print()


def test_sa_rastrigin_with_viz():
    """
    SA na Rastrigin s vizualizací.
    """
    from main import rastrigin
    
    print("=" * 60)
    print("SIMULATED ANNEALING - Rastrigin funkce (2D)")
    print("=" * 60)
    
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    
    best_x, best_f = simulated_annealing(
        objective=rastrigin,
        bounds=bounds,
        max_iter=1500,
        T_initial=100.0,
        T_min=0.001,
        alpha=0.96,
        step_sigma=0.8,
        seed=999,
        visualize=True,
        num_points=180,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print()


if __name__ == "__main__":
    # Hlavní vizualizace na Ackley
    test_sa_ackley_with_visualization()
    
    # Rychlý test bez vizualizace
    test_sa_sphere()
    
    # Další vizualizace na Rastrigin (volitelně)
    # test_sa_rastrigin_with_viz()
    
    print("=" * 60)
    print("VŠECHNY TESTY DOKONČENY")
    print("=" * 60)
