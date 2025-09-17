import sys
import os
import math

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import blind_search, sphere, visualize_blind_search_2d, get_default_bounds


def run_tests() -> bool:
    """
    Testy pro slepé vyhledávání.
    """
    # Parametry problému
    bounds = [(-5.12, 5.12)] * 2  # 2D sphere

    # Spustíme s pevným seedem pro reprodukovatelnost
    best_x, best_f = blind_search(sphere, bounds, npop=50, g_max=50, seed=123)

    print(f"Výsledek blind_search: x={best_x}, f(x)={best_f}")

    # Očekáváme, že f(x) bude poměrně malé (není garance optimálního řešení, jen sanity check)
    ok = best_f < 0.5
    print(f"Kontrola: best_f < 0.5 => {'PASS' if ok else 'FAIL'}")

    return ok


if __name__ == "__main__":
    # Volitelná vizualizace: slepé vyhledávání nad 2D výřezem Sphere
    bounds2d = get_default_bounds("sphere", 2)
    visualize_blind_search_2d(sphere, bounds_2d=bounds2d, npop=40, g_max=40, seed=123, num_points=60, pause_seconds=0.03)
    print("Spouštím testy pro Blind Search...\n")
    success = run_tests()
    if success:
        print("\nTest blind_search prošel.")
        sys.exit(0)
    else:
        print("\nTest blind_search selhal.")
        sys.exit(1)
