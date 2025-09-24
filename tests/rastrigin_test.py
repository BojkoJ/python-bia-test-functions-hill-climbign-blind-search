import sys
import os 
import math

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import rastrigin, plot_surface_2d, get_default_bounds  # Import funkce a vizualizace


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    """
    Porovnání reálných čísel s tolerancí.
    """
    return abs(a - b) <= tol


def run_tests() -> bool:
    """
    Jednoduché testy pro funkci Rastrigin.
    """
    cases = [
        ([], 0.0),                # n=0 -> 10*n + 0 = 0
        ([0.0], 0.0),             # 10*1 + (0 - 10*cos(0)) = 10 - 10 = 0
        ([0.0, 0.0], 0.0),
        ([1.0], 10.0 + (1.0**2 - 10*math.cos(2*math.pi*1.0))),
        ([1.0, -1.0], 20.0 + ((1.0**2 - 10*math.cos(2*math.pi*1.0)) + ((-1.0)**2 - 10*math.cos(2*math.pi*(-1.0))))),
        ([0.5], 10.0 + (0.25 - 10*math.cos(2*math.pi*0.5))),
    ]

    passed = True
    for i, (params, expected) in enumerate(cases, start=1):
        result = rastrigin(params)
        ok = almost_equal(result, expected)
        print(f"Test Rastrigin {i}: vstup={params} -> výsledek={result}, očekávané={expected} => {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False



    return passed


if __name__ == "__main__":
    bounds = get_default_bounds("rastrigin", 2)
    plot_surface_2d(rastrigin, bounds_2d=bounds, num_points=80, title="Rastrigin – 2D výřez")
    print("Spouštím testy pro Rastrigin...\n")
    success = run_tests()
    if success:
        print("\nVšechny testy Rastrigin prošly.")
        sys.exit(0)
    else:
        print("\nNěkteré testy Rastrigin selhaly.")
        sys.exit(1)
