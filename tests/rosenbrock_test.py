import sys 
import os  

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import rosenbrock, plot_surface_2d, get_default_bounds  # Import funkce a vizualizace


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    """
    Porovnání reálných čísel s tolerancí (u Rosenbrocka stačí malá tolerance).
    """
    return abs(a - b) <= tol


def run_tests() -> bool:
    """
    Jednoduché testy pro funkci Rosenbrock.
    """
    cases = [
        ([], 0.0),               # n = 0 -> součet prázdný
        ([1.0], 0.0),            # n = 1 -> součet prázdný
        ([1.0, 1.0], 0.0),       # globální minimum
        ([0.0, 0.0], (1 - 0.0)**2 + 100*(0.0 - 0.0**2)**2),
        ([1.0, 2.0], 100*(2.0 - 1.0**2)**2 + (1 - 1.0)**2),
        ([1.0, 1.0, 1.0], 0.0),  # globální minimum ve 3D
        ([1.2, 1.2], 100*(1.2 - 1.2**2)**2 + (1 - 1.2)**2),
    ]

    passed = True
    for i, (params, expected) in enumerate(cases, start=1):
        result = rosenbrock(params)
        ok = almost_equal(result, expected)
        print(f"Test Rosenbrock {i}: vstup={params} -> výsledek={result}, očekávané={expected} => {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    return passed


if __name__ == "__main__":
    # Interaktivní 3D výřez
    bounds = get_default_bounds("rosenbrock", 2)
    plot_surface_2d(rosenbrock, bounds_2d=bounds, num_points=80, title="Rosenbrock – 2D výřez")
    print("Spouštím testy pro Rosenbrock...\n")
    success = run_tests()
    if success:
        print("\nVšechny testy Rosenbrock prošly.")
        sys.exit(0)
    else:
        print("\nNěkteré testy Rosenbrock selhaly.")
        sys.exit(1)
