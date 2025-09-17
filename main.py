import numpy as np
import math
from typing import Callable, List, Tuple, Optional

# TODO: Vizualizace 3D
# TODO: Pro každou funkci jiné hranice (výřez z funkce) (lower a upper bound)
# Funkce dostane 2 parametry, vyplivne třetí
# TODO: Dělat tak, ať se dají ty parametry vytáhnou ven
# Použít numpy na usnadnění implementace a vizualizaci pomocí matplotlib, taky lze použít na většinu výpočtu
# Snažit se vyhnout čistému iterováni v Pythonu
# Ten algoritmus co prohledává prostor jako takový (např. blind search) nesmí vyletět mimo hranice, když narazí na hranici - prochází okolo hranice a ne přes ni
# Nesmí prostě vyletět mimo ten výřez funkce
# Volit hranice tak, aby každá funkce vypadala ve vizualizaci jinak (jiný výřez u funkcí)
# Všechny solutiony pro 3 dimenze

# ------------------------------------------------------------
# Pomocné nástroje pro hranice, mřížku a 3D vizualizaci
# ------------------------------------------------------------

# Výchozí hranice (výřezy) pro 2D vizualizaci (X,Y). Voleny tak, aby byl tvar zajímavý a odlišný.
DEFAULT_BOUNDS_2D = {
    "sphere": (-5.0, 5.0),          # pro obě osy stejné
    "schwefel": (-500.0, 500.0),    # klasická doména (pozor na měřítko, je hrubší)
    "rosenbrock": (-2.0, 2.0),      # úzké údolí, viditelné při menším rozsahu
    "rastrigin": (-5.12, 5.12),
    "griewank": (-6.0, 6.0),
    "levy": (-3.0, 3.0),
    "michalewicz": (0.0, float(math.pi)),  # [0, π]
    "zakharov": (-5.0, 5.0),
    "ackley": (-5.0, 5.0),
}

def get_default_bounds(func_name: str, dim: int = 2) -> List[Tuple[float, float]]:
    """
    Vrátí výchozí hranice pro danou funkci a dimenzi.
    Pro 2D vrací [(low, high), (low, high)]; pro vyšší dimenze opakuje tentýž interval.
    """
    low, high = DEFAULT_BOUNDS_2D.get(func_name.lower(), (-5.0, 5.0))
    return [(low, high) for _ in range(dim)]


def generate_grid(bounds_2d: List[Tuple[float, float]], num_points: int = 100):
    """
    Vytvoří mřížku (X, Y) v zadaných 2D hranicích. Každá z os má `num_points` vzorků.
    """
    (x_min, x_max), (y_min, y_max) = bounds_2d
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)
    return X, Y


def evaluate_surface_2d(objective: Callable[[List[float]], float],
                        bounds_2d: List[Tuple[float, float]],
                        num_points: int = 100):
    """
    Vyhodnotí funkci na 2D mřížce: funkce dostane 2 parametry (x, y) a vrátí třetí (z).
    Používáme numpy.vectorize pro čitelnost (není to nejrychlejší, ale je přehledné).
    """
    X, Y = generate_grid(bounds_2d, num_points)
    # vektorová aplikace objective na páry (x,y)
    f_vec = np.vectorize(lambda x, y: objective([float(x), float(y)]))
    Z = f_vec(X, Y)
    return X, Y, Z


def plot_surface_2d(objective: Callable[[List[float]], float],
                    bounds_2d: Optional[List[Tuple[float, float]]] = None,
                    num_points: int = 100,
                    title: Optional[str] = None):
    """
    Jednoduchá 3D vizualizace povrchu funkce (2 vstupy -> 3D graf X,Y,Z).
    Importujeme matplotlib až zde, aby to nebylo povinné při běžných výpočtech.
    """
    if bounds_2d is None:
        bounds_2d = get_default_bounds("sphere", 2)  # výchozí: sphere rozsah

    X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, potřebné pro 3D

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        X,
        Y,
        Z,
        cmap='jet',
        edgecolor='k',     # černé hrany (edges)
        linewidth=0.2,     # tenké čáry hran
        antialiased=True,  
        rstride=1,         # vykreslit každou řádku mřížky
        cstride=1,         # vykreslit každý sloupec mřížky
    )
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def find_min_on_grid(objective: Callable[[List[float]], float],
                     bounds_2d: List[Tuple[float, float]],
                     num_points: int = 100) -> Tuple[float, float, float]:
    """
    Najde přibližné minimum na 2D mřížce a vrátí (x, y, z).
    Je to hrubá metoda na odhad – přesnost závisí na `num_points`.
    """
    X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points)
    # Najdeme index minima v Z a přemapujeme zpět na souřadnice X,Y
    idx = np.unravel_index(np.argmin(Z), Z.shape)
    x_best = float(X[idx])
    y_best = float(Y[idx])
    z_best = float(Z[idx])
    return x_best, y_best, z_best


def visualize_blind_search_2d(
    objective: Callable[[List[float]], float],
    bounds_2d: Optional[List[Tuple[float, float]]] = None,
    npop: int = 30,
    g_max: int = 30,
    seed: Optional[int] = None,
    num_points: int = 150,
    pause_seconds: float = 0.05,
):
    """
    Vizualizace slepého (náhodného) vyhledávání v 2D na 3D povrchu funkce.

    Co uvidíte:
    - Pevný 3D povrch funkce f(x1, x2)
    - V každé generaci se vykreslí náhodní kandidáti (malé tečky)
    - Nejlepší nalezený bod (globálně) je zvýrazněn větší červenou tečkou
    - Průběh nejlepších bodů v čase je zobrazen jako červená křivka (trajektorie)

    Parametry
    ---------
    objective : funkce
        Cílová funkce f(x), která pro [x1, x2] vrací číslo (menší je lepší).
    bounds_2d : list[(float, float)] | None
        Meze pro 2D vizualizaci. Pokud None, zvolí se implicitně (sphere, 2D).
    npop : int
        Počet kandidátů v jedné generaci.
    g_max : int
        Počet generací.
    seed : int | None
        Volitelný seed pro reprodukovatelnost.
    num_points : int
        Hustota mřížky povrchu (čím více, tím hladší povrch; pomalejší vykreslování).
    pause_seconds : float
        Krátká pauza mezi generacemi (ovlivní rychlost animace).
    """
    import random
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if bounds_2d is None:
        bounds_2d = get_default_bounds("sphere", 2)

    # Připravíme povrch funkce
    X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points=num_points)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap='jet',
        edgecolor='k',
        linewidth=0.2,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    ax.set_title('Vizualizace blind search (2D → 3D)')

    (x_min, x_max), (y_min, y_max) = bounds_2d

    rng = random.Random(seed)

    def random_point_2d():
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        # clamp, pro jistotu
        if x < x_min:
            x = x_min
        if x > x_max:
            x = x_max
        if y < y_min:
            y = y_min
        if y > y_max:
            y = y_max
        return x, y

    # Náhodný start
    best_x, best_y = random_point_2d()
    best_f = objective([best_x, best_y])

    # Umělců (artists) několik: kandidáti, nejlepší bod, trajektorie
    candidates_scatter = ax.scatter([], [], [], s=10, c='k', alpha=0.6, depthshade=False)
    best_scatter = ax.scatter([best_x], [best_y], [best_f], s=50, c='red', depthshade=False)
    path_x = [best_x]
    path_y = [best_y]
    path_z = [best_f]
    path_line, = ax.plot(path_x, path_y, path_z, color='red', linewidth=1.5)

    plt.tight_layout()
    plt.pause(0.1)  # malá pauza, aby se figura správně inicializovala

    g = 0
    while g < g_max:
        # Vygenerujeme novou populaci kandidátů
        cand_x = []
        cand_y = []
        cand_z = []
        for _ in range(npop):
            x, y = random_point_2d()
            z = objective([x, y])
            cand_x.append(x)
            cand_y.append(y)
            cand_z.append(z)

        # Aktualizace kandidátů na grafu
        candidates_scatter._offsets3d = (cand_x, cand_y, cand_z)

        # Najdeme nejlepší v této generaci
        best_idx = 0
        best_gen_f = cand_z[0]
        for i in range(1, len(cand_z)):
            if cand_z[i] < best_gen_f:
                best_gen_f = cand_z[i]
                best_idx = i

        # Pokud zlepšuje globální nejlepší, zapíšeme a vykreslíme
        if best_gen_f < best_f:
            best_x = cand_x[best_idx]
            best_y = cand_y[best_idx]
            best_f = best_gen_f
            # posuneme značku nejlepšího bodu
            best_scatter._offsets3d = ([best_x], [best_y], [best_f])
            # aktualizujeme trajektorii
            path_x.append(best_x)
            path_y.append(best_y)
            path_z.append(best_f)
            path_line.set_data(path_x, path_y)
            path_line.set_3d_properties(path_z)

        plt.pause(pause_seconds)
        g += 1

    # Na závěr malé přiblížení a zobrazení
    plt.show()

# Jednotlivé testovací funkce:

def sphere(params):
    """
    Definice:
        f(x) = sum(x_i^2) pro i = 1..n
    
    - n je rozměr (počet prvků vektoru x)
    - Obvyklá doména: x_i v intervalu [-5.12, 5.12] pro všechna i = 1, ..., d
    - Globální minimum: x* = (0, 0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Sphere pro zadané params
    """
    total = 0.0
    for value in params:
        total += value * value
    return float(total)

def schwefel(params):
    """
    Definice:
        f(x) = 418.9829 * n - sum_{i=1..n}  x_i * sin(sqrt(|x_i|))]

    - n je rozměr (počet prvků vektoru x)
    - Obvyklá doména: x_i v intervalu [-500, 500]
    - Globální minimum: x_i ≈ 420.968746... pro všechny i, hodnota f(x*) ≈ 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Schwefel pro zadané params
    """
    n = 0
    suma = 0.0
    for value in params:
        n += 1
        # Použijeme absolutní hodnotu uvnitř odmocniny podle definice.
        term = value * math.sin(math.sqrt(abs(value)))
        suma += term

    konst = 418.9829
    result = konst * n - suma
    return float(result)


def rosenbrock(params):
    """
    Definice:
        f(x) = sum_{i=1..n-1} [ 100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]

    - n je rozměr vektoru x (pro n < 2 je součet prázdný -> 0)
    - Obvyklá doména: x_i v intervalu přibližně [-2.5, 2.5] (často uváděno), někdy [-5, 10]
    - Globální minimum: x* = (1, 1, ..., 1) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Rosenbrock pro zadané params
    """
    total = 0.0
    # Součet jde od i=0 do i=n-2 (tj. pracujeme vždy s dvojicí x_i a x_{i+1})
    n = len(params)
    for i in range(0, n - 1):
        xi = params[i]
        x_next = params[i + 1]
        # 100 * (x_{i+1} - x_i^2)^2
        first = 100.0 * (x_next - (xi * xi)) ** 2
        # (1 - x_i)^2
        second = (1.0 - xi) ** 2
        total += first + second

    return float(total)

def rastrigin(params):
    """
    Definice pro n-rozměrný vektor x:
        f(x) = 10 * n + sum_{i=1..n} [ x_i^2 - 10 * cos(2π x_i) ]

    - Obvyklá doména: x_i v intervalu [-5.12, 5.12]
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Rastrigin pro zadané params
    """
    n = 0
    total = 0.0
    for value in params:
        n += 1
        total += (value * value) - 10.0 * math.cos(2.0 * math.pi * value)

    result = 10.0 * n + total
    return float(result)


def griewank(params):
    r"""
    Definice pro n-rozměrný vektor x:
        f(x) = 1 + \sum_{i=1..n} (x_i^2 / 4000) - \prod_{i=1..n} cos\left(\frac{x_i}{\sqrt{i}}\right)

    - Obvyklá doména: x_i v intervalu [-600, 600]
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Griewank pro zadané `params`.
    """
    sum_term = 0.0
    prod_term = 1.0

    # i číslujeme od 1 kvůli definici s odmocninou i
    i = 1
    for value in params:
        sum_term += (value * value) / 4000.0
        prod_term *= math.cos(value / math.sqrt(i))
        i += 1

    result = 1.0 + sum_term - prod_term
    return float(result)


def levy(params):
    """
    Definice pro n-rozměrný vektor x:
        f(x) = sin^2(3π x_1)
               + sum_{i=1..n-1} (x_i - 1)^2 * (1 + sin^2(3π x_i + 1))
               + (x_n - 1)^2 * (1 + sin^2(2π x_n))

    - Obvyklá doména: x_i v intervalu [-10, 10]
    - Globální minimum: x* = (1, ..., 1) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Lévy pro zadané `params`.
    """
    n = len(params)
    if n == 0:
        return 0.0

    # První člen: sin^2(3π x_1)
    x1 = params[0]
    total = (math.sin(3.0 * math.pi * x1)) ** 2

    # Prostřední součet: i = 1..n-1 (indexy 0..n-2)
    for i in range(0, n - 1):
        xi = params[i]
        total += (xi - 1.0) * (xi - 1.0) * (1.0 + (math.sin(3.0 * math.pi * xi + 1.0) ** 2))

    # Poslední člen: (x_n - 1)^2 * (1 + sin^2(2π x_n))
    xn = params[-1]
    total += (xn - 1.0) * (xn - 1.0) * (1.0 + (math.sin(2.0 * math.pi * xn) ** 2))

    return float(total)


def michalewicz(params):
    r"""
    Definice pro n-rozměrný vektor x (obvykle s parametrem m = 10):
        f(x) = - \sum_{i=1..n} [ sin(x_i) * ( sin( i * x_i^2 / π ) )^{2m} ]

    - Obvyklá doména: x_i v intervalu [0, π]
    - Typické nastavení: m = 10 (čím větší m, tím více lokálních minim)
    - Globální minimum pro n=2, m=10 je přibližně f(x*) ≈ -1.8013 v bodě x* ≈ (2.20, 1.57)

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Michalewicz pro zadané `params` (při m = 10).
    """
    m = 10
    total = 0.0
    i = 1
    for x in params:
        s1 = math.sin(x)
        s2 = math.sin(i * (x * x) / math.pi)
        term = s1 * (s2 ** (2 * m))
        total += term
        i += 1
    return float(-total)


def zakharov(params):
    r"""
    Funkce Zakharov (minimalizační úloha).

    Definice pro n-rozměrný vektor x:
        f(x) = \sum_{i=1..n} x_i^2
               + (\sum_{i=1..n} 0.5 * i * x_i)^2
               + (\sum_{i=1..n} 0.5 * i * x_i)^4

    - Obvyklá doména: x_i v intervalu [-5, 10] (různé zdroje uvádí mírně odlišně)
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Zakharov pro zadané `params`.
    """
    sum_sq = 0.0
    sum_lin = 0.0
    i = 1
    for x in params:
        sum_sq += x * x
        sum_lin += 0.5 * i * x
        i += 1

    result = sum_sq + (sum_lin ** 2) + (sum_lin ** 4)
    return float(result)


def ackley(params):
    """
    Funkce Ackley (minimalizační úloha).

    Pro n-rozměrný vektor x a konstanty a=20, b=0.2, c=2π:
        f(x) = -a * exp(-b * sqrt( (1/n) * sum(x_i^2) ))
               - exp( (1/n) * sum( cos(c * x_i) ) )
               + a + e

    - Obvyklá doména: x_i v intervalu [-32.768, 32.768]
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Ackley pro zadané `params`.
    """
    n = 0
    sum_sq = 0.0
    sum_cos = 0.0
    for x in params:
        n += 1
        sum_sq += x * x
        sum_cos += math.cos(2.0 * math.pi * x)

    if n == 0:
        return 0.0

    a = 20.0
    b = 0.2
    # c = 2*pi je použito přímo v cyklu výše

    term1 = -a * math.exp(-b * math.sqrt((1.0 / n) * sum_sq))
    term2 = -math.exp((1.0 / n) * sum_cos)
    result = term1 + term2 + a + math.e
    return float(result)


def blind_search(
    objective: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    npop: int,
    g_max: int,
    seed: Optional[int] = None,
) -> Tuple[List[float], float]:
    """
    Slepé (náhodné) vyhledávání pro minimalizační úlohu.

    Popis:
    - vygenerujeme náhodné počáteční řešení x_b
    - opakovaně (max g_max generací) vytvoříme NP náhodných kandidátů v mezích
    - vyhodnotíme je a pokud najdeme lepší řešení x_s, nahradíme jím x_b
    - na konci vrátíme nejlepší nalezené řešení a jeho hodnotu

    Parametry
    ---------
    objective : funkce
        Cílová funkce f(x), která pro zadaný vektor vrací číslo (menší je lepší).
    bounds : seznam dvojic (min_i, max_i)
        Meze pro každou souřadnici řešení.
    npop : int
        Počet náhodných kandidátů v jedné generaci.
    g_max : int
        Maximální počet generací.
    seed : int nebo None
        Volitelný seed pro reprodukovatelnost.

    Návratová hodnota
    -----------------
    (best_x, best_f) : (list, float)
        Nejlepší nalezený vektor a jeho hodnota funkce.
    """
    import random

    rng = random.Random(seed)

    # Pomocná funkce: vygeneruje jeden náhodný vektor v mezích
    def random_vector():
        vec = []
        for low, high in bounds:
            value = rng.uniform(low, high)
            # jistota, že nevyjedeme mimo hranice (clamp)
            if value < low:
                value = low
            if value > high:
                value = high
            vec.append(value)
        return vec

    # 1) náhodné počáteční řešení
    best_x = random_vector()
    best_f = objective(best_x)

    # 2) hlavní smyčka přes generace
    g = 0
    while g < g_max:
        # 3) vygenerujeme NP kandidátů
        candidates = []
        for _ in range(npop):
            candidates.append(random_vector())

        # 4) vyhodnotíme kandidáty a najdeme nejlepší v této generaci
        best_x_gen = None
        best_f_gen = None
        for x in candidates:
            fx = objective(x)
            if (best_f_gen is None) or (fx < best_f_gen):
                best_x_gen = x
                best_f_gen = fx

        # 5) pokud zlepšuje globální nejlepší, nahradíme
        if best_f_gen < best_f:
            best_x = best_x_gen
            best_f = best_f_gen

        # 6) další generace
        g += 1

    return best_x, float(best_f)
