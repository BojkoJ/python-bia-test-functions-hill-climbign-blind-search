import sys  # Modul Pythonu pro práci s běžícím programem (např. návratové kódy, sys.path)
import os   # Modul Pythonu pro práci s cestami a adresáři

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import zakharov, plot_surface_2d, get_default_bounds  # Import testované funkce a vizualizace


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def run_tests() -> bool:
    """
    Jednoduché testy pro funkci Zakharov.
    """
    cases = [
        ([], 0.0),                   # prázdný vektor -> 0
        ([0.0], 0.0),                # minimum v nule (1D)
        ([0.0, 0.0], 0.0),           # minimum v nule (2D)
        ([1.0], 1.0 + (0.5*1*1.0)**2 + (0.5*1*1.0)**4),
        ([1.0, 2.0], (1.0**2 + 2.0**2) + (0.5*1*1.0 + 0.5*2*2.0)**2 + (0.5*1*1.0 + 0.5*2*2.0)**4),
    ]

    passed = True
    for i, (params, expected) in enumerate(cases, start=1):
        result = zakharov(params)
        ok = almost_equal(result, expected)
        print(f"Test Zakharov {i}: vstup={params} -> výsledek={result}, očekávané={expected} => {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    # Kontrola minima ve 3D v nule
    zero3 = [0.0, 0.0, 0.0]
    val_zero3 = zakharov(zero3)
    ok = almost_equal(val_zero3, 0.0)
    print(f"Test Zakharov (minimum v nule, 3D): x={zero3} -> f(x)={val_zero3} => {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    return passed


if __name__ == "__main__":
    # Interaktivní 3D výřez
    bounds = get_default_bounds("zakharov", 2)
    plot_surface_2d(zakharov, bounds_2d=bounds, num_points=80, title="Zakharov – 2D výřez")
    print("Spouštím testy pro Zakharov...\n")
    success = run_tests()
    if success:
        print("\nVšechny testy Zakharov prošly.")
        sys.exit(0)
    else:
        print("\nNěkteré testy Zakharov selhaly.")
        sys.exit(1)
