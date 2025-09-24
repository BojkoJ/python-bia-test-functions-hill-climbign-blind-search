import sys 
import os  
import math

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import levy, plot_surface_2d, get_default_bounds  # Import standardní Levy funkce a vizualizace


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    """
    Porovnání reálných čísel s tolerancí.
    """
    return abs(a - b) <= tol


def levy_manual_eval(x_list):
    """Pomocná funkce: ruční výpočet standardní Levy přes definici (pro ověření testů)."""
    # w transformace
    w = [1.0 + (x - 1.0)/4.0 for x in x_list]
    n = len(w)
    if n == 0:
        return 0.0
    total = math.sin(math.pi * w[0]) ** 2
    for i in range(0, n-1):
        wi = w[i]
        total += (wi - 1.0)**2 * (1.0 + 10.0 * (math.sin(math.pi * wi + 1.0)**2))
    wn = w[-1]
    total += (wn - 1.0)**2 * (1.0 + (math.sin(2.0 * math.pi * wn)**2))
    return total


def run_tests() -> bool:
    """
    Testy pro standardní Levy funkci (s w transformací).
    """
    cases = [
        ([], 0.0),
        ([1.0], 0.0),               # minimum
        ([1.0, 1.0], 0.0),           # minimum
        ([1.0, 1.0, 1.0], 0.0),      # minimum ve 3D
        ([0.0], levy_manual_eval([0.0])),
        ([2.0], levy_manual_eval([2.0])),
        ([0.0, 2.0], levy_manual_eval([0.0, 2.0])),
        ([1.0, 2.0], levy_manual_eval([1.0, 2.0])),
    ]

    passed = True
    for i, (params, expected) in enumerate(cases, start=1):
        result = levy(params)
        ok = almost_equal(result, expected)
        print(f"Test Levy {i}: vstup={params} -> výsledek={result}, očekávané={expected} => {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    # Extra: náhodná kontrola konzistence pro několik náhodných bodů (deterministicky)
    import random
    rng = random.Random(123)
    for j in range(3):
        trial = [rng.uniform(-5, 5), rng.uniform(-5, 5)]
        expected = levy_manual_eval(trial)
        result = levy(trial)
        ok = almost_equal(result, expected, tol=1e-9)
        print(f"Kontrola náhodný bod {j+1}: x={trial} -> f={result}, manuál={expected} => {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    return passed


if __name__ == "__main__":
    # Zobrazíme 3D výřez pro lepší představu o tvaru funkce
    bounds = get_default_bounds("levy", 2)
    plot_surface_2d(levy, bounds_2d=bounds, num_points=160, title="Levy – standardní definice")
    print("Spouštím testy pro Lévy...\n")
    success = run_tests()
    if success:
        print("\nVšechny testy Lévy prošly.")
        sys.exit(0)
    else:
        print("\nNěkteré testy Lévy selhaly.")
        sys.exit(1)
