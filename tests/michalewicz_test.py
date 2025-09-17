import sys  # Modul Pythonu pro práci s běžícím programem (např. návratové kódy, sys.path)
import os   # Modul Pythonu pro práci s cestami a adresáři
import math

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import michalewicz, plot_surface_2d, get_default_bounds  # Import testované funkce a vizualizace


def almost_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    """
    Porovnání reálných čísel s tolerancí (u Michalewicze ponecháme vyšší toleranci).
    """
    return abs(a - b) <= tol


def run_tests() -> bool:
    """
    Jednoduché testy pro funkci Michalewicz (m=10).
    """
    cases = [
        ([], 0.0),
        ([0.0], -0.0),  # sin(0)=0 -> výsledek 0
        ([math.pi], -0.0),
        ([math.pi/2], -math.sin(math.pi/2) * (math.sin(1 * ((math.pi/2)**2) / math.pi) ** (2*10))),
        ([math.pi/2, math.pi/2],
         -(
             math.sin(math.pi/2) * (math.sin(1 * ((math.pi/2)**2) / math.pi) ** (2*10))
             + math.sin(math.pi/2) * (math.sin(2 * ((math.pi/2)**2) / math.pi) ** (2*10))
          )
        ),
    ]

    passed = True
    for i, (params, expected) in enumerate(cases, start=1):
        result = michalewicz(params)
        ok = almost_equal(result, expected)
        print(f"Test Michalewicz {i}: vstup={params} -> výsledek={result}, očekávané={expected} => {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    # Kontrola známé hodnoty pro n=2, m=10 (přibližná)
    x_star = [2.20, 1.57]
    val_star = michalewicz(x_star)
    ok = val_star < -1.7  # hrubá dolní hranice pro kontrolu tvaru funkce
    print(f"Test Michalewicz (n≈2 optimum): x={x_star} -> f(x)={val_star} (< -1.7?) => {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    return passed


if __name__ == "__main__":
    # Interaktivní 3D výřez (doména [0, π])
    bounds = get_default_bounds("michalewicz", 2)
    plot_surface_2d(michalewicz, bounds_2d=bounds, num_points=80, title="Michalewicz – 2D výřez")
    print("Spouštím testy pro Michalewicz...\n")
    success = run_tests()
    if success:
        print("\nVšechny testy Michalewicz prošly.")
        sys.exit(0)
    else:
        print("\nNěkteré testy Michalewicz selhaly.")
        sys.exit(1)
