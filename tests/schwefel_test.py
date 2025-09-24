import sys  
import os   
import math

# Nastavení cesty pro import z kořene projektu (aby šel importovat soubor main.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import schwefel, plot_surface_2d, get_default_bounds  # Import testované funkce a vizualizace


def almost_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    """
    Porovnání reálných čísel s tolerancí.
    Použijeme o něco vyšší toleranci než u Sphere, protože se používá sin a odmocnina.
    """
    return abs(a - b) <= tol


def run_tests() -> bool:
    """
    Jednoduché testy pro funkci Schwefel.
    """
    cases = [
        # Základní okrajové případy
        ([], 0.0),                 # n = 0 -> 418.9829 * 0 - 0 = 0
        ([0.0], 418.9829),         # 418.9829*1 - (0*sin(sqrt(0))) = 418.9829
        ([0.0, 0.0], 2 * 418.9829),

        # Jednoduché hodnoty
        ([1.0], 418.9829 - (1.0 * math.sin(math.sqrt(1.0)))),
        ([-1.0], 418.9829 - (-1.0 * math.sin(math.sqrt(1.0)))),

        # Více rozměrů
        ([1.0, 2.0], 2*418.9829 - (1.0*math.sin(math.sqrt(1.0)) + 2.0*math.sin(math.sqrt(2.0)))),
    ]

    passed = True
    for i, (params, expected) in enumerate(cases, start=1):
        result = schwefel(params)
        ok = almost_equal(result, expected)
        print(f"Test Schwefel {i}: vstup={params} -> výsledek={result}, očekávané={expected} => {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    

    return passed


if __name__ == "__main__":
    # Nejdříve zobrazíme 3D výřez funkce (lze otáčet myší)
    bounds = get_default_bounds("schwefel", 2)
    plot_surface_2d(schwefel, bounds_2d=bounds, num_points=100, title="Schwefel – 2D výřez")
    print("Spouštím testy pro Schwefel...\n")
    success = run_tests()
    if success:
        print("\nVšechny testy Schwefel prošly.")
        sys.exit(0)
    else:
        print("\nNěkteré testy Schwefel selhaly.")
        sys.exit(1)
