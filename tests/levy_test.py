import sys  # Modul Pythonu pro práci s běžícím programem (např. návratové kódy, sys.path)
import os   # Modul Pythonu pro práci s cestami a adresáři
import math

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import levy, plot_surface_2d, get_default_bounds  # Import testované funkce a vizualizace


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    """
    Porovnání reálných čísel s tolerancí.
    """
    return abs(a - b) <= tol


def run_tests() -> bool:
    """
    Jednoduché testy pro základní funkci Lévy.
    """
    cases = [
        ([], 0.0),  # prázdný vektor -> 0
        ([1.0], 0.0),  # globální minimum (1D)
        ([1.0, 1.0], 0.0),  # globální minimum (2D)
        # Následují ruční výpočty podle základní definice
        ([0.0], (math.sin(3*math.pi*0.0))**2 + (0.0 - 1.0)**2 * (1 + math.sin(2*math.pi*0.0)**2)),
        ([2.0], (math.sin(3*math.pi*2.0))**2 + (2.0 - 1.0)**2 * (1 + math.sin(2*math.pi*2.0)**2)),
        ([1.0, 2.0],
        (math.sin(3*math.pi*1.0))**2 + (1.0 - 1.0)**2 * (1 + (math.sin(3*math.pi*1.0 + 1.0))**2) 
        + (2.0 - 1.0)**2 * (1 + (math.sin(2*math.pi*2.0))**2)
        ),
    ]

    passed = True
    for i, (params, expected) in enumerate(cases, start=1):
        result = levy(params)
        ok = almost_equal(result, expected)
        print(f"Test Lévy {i}: vstup={params} -> výsledek={result}, očekávané={expected} => {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    # Kontrola minima v bodě (1,1,1)
    ones3 = [1.0, 1.0, 1.0]
    val_ones3 = levy(ones3)
    ok = almost_equal(val_ones3, 0.0)
    print(f"Test Lévy (minimum v (1,1,1)): x={ones3} -> f(x)={val_ones3} => {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    return passed


if __name__ == "__main__":
    # Zobrazíme 3D výřez pro lepší představu o tvaru funkce
    bounds = get_default_bounds("levy", 2)
    plot_surface_2d(levy, bounds_2d=bounds, num_points=80, title="Lévy – 2D výřez")
    print("Spouštím testy pro Lévy...\n")
    success = run_tests()
    if success:
        print("\nVšechny testy Lévy prošly.")
        sys.exit(0)
    else:
        print("\nNěkteré testy Lévy selhaly.")
        sys.exit(1)
