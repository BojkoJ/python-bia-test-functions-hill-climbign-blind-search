import sys  # Modul pro práci s běžícím programem (např. návratové kódy, sys.path)
import os   # Modul pro práci s cestami a adresáři

# Chceme importovat funkci `sphere` ze souboru `main.py`, který je o složku výše.
# Když spustíme tento test přímo (python tests/sphere_test.py), Pythonu musíme pomoci najít `main.py`.
# Následující řádky spočítají cestu ke kořeni projektu a přidají ji do vyhledávací cesty modulů (sys.path).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # Díky tomu bude `import main` fungovat i při spuštění ze složky tests

from main import sphere, plot_surface_2d, get_default_bounds  # Import funkce a nástrojů pro vizualizaci

def almost_equal(a: float, b: float, tol: float = 0.000000001) -> bool:
    """
    Vrátí True, pokud se čísla a a b liší nejvýše o `tol`.
    Použít pro bezpečné porovnání desetinných čísel.

    U reálných čísel může kvůli zaokrouhlování vzniknout drobná odchylka, 
    takže přímé porovnání a == b nemusí být vždy spolehlivé.
    """
    return abs(a - b) <= tol


def run_tests() -> bool:
    """
    Spustí sadu testů pro funkci Sphere.
    True, pokud všechny testy projdou, jinak False.
    """

    # Testovací případy jsou tuply: (vstupní_vektor, očekávaná_hodnota)
    # Příklad: pro [1, 2, 3] je Sphere 1^2 + 2^2 + 3^2 = 14
    cases = [
        ([], 0.0),                      # Prázdný seznam = 0
        ([0], 0.0),                     # 0^2 = 0
        ([0, 0], 0.0),                  # 0^2 + 0^2 = 0
        ([1, 2, 3], 14.0),              # 1 + 4 + 9 = 14
        ([-1, -2], 5.0),                # (-1)^2 + (-2)^2 = 1 + 4 = 5
        ([0.5, -0.5], 0.5),             # 0.25 + 0.25 = 0.5
        ([1.0, -1.0, 1.0, -1.0], 4.0),  # 4
    ]

    passed = True  # Předpokládáme, že všechny testy projdou; pokud nějaký selže, nastavíme na False

    for i, (params, expected) in enumerate(cases, start=1):
        result = sphere(params)                # Zavoláme testovanou funkci
        ok = almost_equal(result, expected)    # Porovnáme s očekávanou hodnotou (s tolerancí)

        print(
            f"Test Sphere {i}: vstup={params} -> výsledek={result}, očekávané={expected} => {'PASS' if ok else 'FAIL'}"
        )

        # Pokud výsledek neodpovídá, test neprošel
        if not ok:
            passed = False

    return passed

if __name__ == "__main__":
    # Nejdříve zobrazíme 3D výřez funkce (lze otáčet myší)
    bounds = get_default_bounds("sphere", 2)
    plot_surface_2d(sphere, bounds_2d=bounds, num_points=80, title="Sphere – 2D výřez")

    print("Spouštím testy pro Sphere...\n")
    success = run_tests()

    if success:
        print("\nVšechny testy Sphere prošly.")
        sys.exit(0)
    else:
        print("\nNěkteré testy Sphere selhaly.")
        sys.exit(1)
