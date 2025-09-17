import sys  # Modul Pythonu pro práci s běžícím programem (např. návratové kódy, sys.path)
import os   # Modul Pythonu pro práci s cestami a adresáři
import math

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import ackley, plot_surface_2d, get_default_bounds  # Import testované funkce a vizualizace


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def run_tests() -> bool:
    """
    Jednoduché testy pro funkci Ackley.
    """
    cases = [
        ([], 0.0),                # n=0 -> definujme 0 (triviální)
        ([0.0], 0.0),             # globální minimum v nule
        ([0.0, 0.0], 0.0),
        ([1.0],
         # ruční výpočet podle definice s a=20, b=0.2, c=2*pi
         -20.0 * math.exp(-0.2 * math.sqrt((1.0/1) * (1.0**2)))
         - math.exp((1.0/1) * math.cos(2*math.pi*1.0))
         + 20.0 + math.e
        ),
        ([1.0, 2.0],
         -20.0 * math.exp(-0.2 * math.sqrt((1.0/2) * (1.0**2 + 2.0**2)))
         - math.exp((1.0/2) * (math.cos(2*math.pi*1.0) + math.cos(2*math.pi*2.0)))
         + 20.0 + math.e
        ),
    ]

    passed = True
    for i, (params, expected) in enumerate(cases, start=1):
        result = ackley(params)
        ok = almost_equal(result, expected)
        print(f"Test Ackley {i}: vstup={params} -> výsledek={result}, očekávané={expected} => {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    # Kontrola minima v nule (3D)
    zero3 = [0.0, 0.0, 0.0]
    val_zero3 = ackley(zero3)
    ok = almost_equal(val_zero3, 0.0)
    print(f"Test Ackley (minimum v nule, 3D): x={zero3} -> f(x)={val_zero3} => {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    return passed


if __name__ == "__main__":
    bounds = get_default_bounds("ackley", 2)
    plot_surface_2d(ackley, bounds_2d=bounds, num_points=80, title="Ackley – 2D výřez")
    print("Spouštím testy pro Ackley...\n")
    success = run_tests()
    if success:
        print("\nVšechny testy Ackley prošly.")
        sys.exit(0)
    else:
        print("\nNěkteré testy Ackley selhaly.")
        sys.exit(1)
