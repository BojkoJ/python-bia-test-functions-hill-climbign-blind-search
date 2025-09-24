import sys
import os  
import math

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import griewank, plot_surface_2d, get_default_bounds  # Import testované funkce a vizualizace


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    """
    Porovnání reálných čísel s tolerancí.
    """
    return abs(a - b) <= tol


def run_tests() -> bool:
    """
    Jednoduché testy pro funkci Griewank.
    """
    cases = [
        ([], 0.0),                # n=0 -> 1 + 0 - 1 = 0
        ([0.0], 0.0),             # 1 + 0 - cos(0) = 0
        ([0.0, 0.0], 0.0),
        ([1.0], 1.0 + (1.0**2)/4000.0 - math.cos(1.0/math.sqrt(1))),
        ([1.0, 2.0], 1.0 + ((1.0**2)/4000.0 + (2.0**2)/4000.0) - (math.cos(1.0/math.sqrt(1)) * math.cos(2.0/math.sqrt(2)))),
    ]

    passed = True
    for i, (params, expected) in enumerate(cases, start=1):
        result = griewank(params)
        ok = almost_equal(result, expected)
        print(f"Test Griewank {i}: vstup={params} -> výsledek={result}, očekávané={expected} => {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    return passed


if __name__ == "__main__":
    # Interaktivní 3D výřez
    bounds = get_default_bounds("griewank", 2)
    plot_surface_2d(griewank, bounds_2d=bounds, num_points=80, title="Griewank – 2D výřez")
    print("Spouštím testy pro Griewank...\n")
    success = run_tests()
    if success:
        print("\nVšechny testy Griewank prošly.")
        sys.exit(0)
    else:
        print("\nNěkteré testy Griewank selhaly.")
        sys.exit(1)
