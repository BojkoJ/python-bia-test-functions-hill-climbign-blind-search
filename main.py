import numpy as np
import math
from typing import Callable, List, Tuple, Optional

# Výchozí hranice (výřezy) pro 2D vizualizaci (X,Y).
DEFAULT_BOUNDS_2D = {
    "sphere": (-5.0, 5.0),          
    "schwefel": (-500.0, 500.0),    
    "rosenbrock": (-5.0, 10.0),   
    "rastrigin": (-5.12, 5.12),
    "griewank": (-6.0, 6.0),
    "levy": (-3.0, 3.0),
    "michalewicz": (0.0, float(math.pi)), 
    "zakharov": (-5.0, 5.0),
    "ackley": (-32.768, 32.768),
}

def get_default_bounds(func_name: str, dim: int = 2) -> List[Tuple[float, float]]:
        """
        Vrátí seznam hranic pro každou dimenzi.

        Parametry:
            func_name : název funkce
            dim       : kolik dimenzí (např. 2 > chceme dvě dvojice hranic).

        Návrat:
            list dvojic (low, high). Každá dvojice jsou float hodnoty dolní a horní meze.
        """
        # Převedeme název na malá písmena
        name_lower = func_name.lower()

        # Zkusíme v tabulce DEFAULT_BOUNDS_2D najít položku podle klíče name_lower, pokud není dáme default
        bounds_pair = DEFAULT_BOUNDS_2D.get(name_lower, (-5.0, 5.0))

        # Rozbalíme dvojici (low, high) do dvou proměnných pro přehlednost.
        # Příklad: bounds_pair = (-5.0, 5.0) -> low = -5.0, high = 5.0
        low = bounds_pair[0]
        high = bounds_pair[1]

        result: List[Tuple[float, float]] = []

        for i in range(dim):
                result.append((low, high))

        return result


def generate_grid(bounds_2d: List[Tuple[float, float]], num_points: int = 100):
    """
    Vytvoří mřížku (X, Y) v zadaných 2D hranicích. Každá z os má `num_points` vzorků.
    """
    (x_min, x_max), (y_min, y_max) = bounds_2d
    
    # np.linspace vytvoří jednorozměrné pole s rovnoměrně rozloženými body
    # mezi zadanými hranicemi. Např. linspace(0, 10, 5) → [0, 2.5, 5, 7.5, 10]
    x = np.linspace(x_min, x_max, num_points) # Vytvoří num_points bodů na x-ose
    y = np.linspace(y_min, y_max, num_points) # Vytvoří num_points bodů na y-ose
    
    # np.meshgrid vezme dva 1D vektory a vytvoří z nich 2D mřížku souřadnic
    # X obsahuje x-souřadnice pro každý bod mřížky
    # Y obsahuje y-souřadnice pro každý bod mřížky
    # Výsledek: každý bod [X[i,j], Y[i,j]] reprezentuje jeden bod v 2D mřížce
    X, Y = np.meshgrid(x, y)
    return X, Y

def evaluate_surface_2d(objective: Callable[[List[float]], float],
                            bounds_2d: List[Tuple[float, float]],
                            num_points: int = 100):
        """
        Vyhodnotí funkci na 2D mřížce: funkce dostane 2 parametry (x, y) a vrátí třetí (z).
        Používáme numpy.vectorize pro čitelnost.
        """
        # Vytvoříme 2D mřížku bodů (X, Y) v zadaných hranicích
        X, Y = generate_grid(bounds_2d, num_points)
        
        # np.vectorize umožňuje aplikovat obyčejnou funkci na numpy pole element po elementu
        # lambda x, y: objective([float(x), float(y)]) převede každou dvojici (x,y) z mřížky
        # na seznam [x, y] a předá ho cílové funkci
        f_vec = np.vectorize(lambda x, y: objective([float(x), float(y)]))
        
        # Aplikujeme vektorizovanou funkci na celé pole X a Y najednou
        # Výsledek Z obsahuje hodnotu funkce pro každý bod mřížky
        # Z[i,j] = objective([X[i,j], Y[i,j]])
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
    if bounds_2d is None:                               # Pokud uživatel nezadal vlastní hranice tak default
        bounds_2d = get_default_bounds("sphere", 2)

    # Vyhodnotíme funkci na pravidelné mřížce bodů (dostaneme X, Y souřadnice a Z hodnoty funkce)
    X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points)

    import matplotlib.pyplot as plt                      # Import knihovny pro vykreslování

    fig = plt.figure(figsize=(8, 6))                     # Vytvoří novou figuru s danou velikostí
    ax = fig.add_subplot(111, projection='3d')           # Přidá 3D subplot

    # Samotné vykreslení 3D povrchu (surface). Jednotlivé parametry určují barvy a mřížku.
    ax.plot_surface(
        X,                                             # 2D pole x souřadnic
        Y,                                             # 2D pole y souřadnic
        Z,                                             # 2D pole hodnot funkce f(x,y)
        cmap='jet',                                    # Barevná mapa
        edgecolor='k',                                 # Černé hrany každého malého polygonu
        linewidth=0.2,                                 # Tloušťka čar hran
        antialiased=True,                              # Vyhlazení hran pro hezčí vzhled
        rstride=1,                                     # Vykreslit každou řádku mřížky
        cstride=1,                                     # Vykreslit každý sloupec mřížky
    )
    
    ax.set_xlabel('x1')                                 # Popisek osy X
    ax.set_ylabel('x2')                                 # Popisek osy Y
    ax.set_zlabel('f(x)')                               # Popisek osy Z
    
    if title:                                           # Pokud je předán titulek nastavíme ho nad graf
        ax.set_title(title)                             

    plt.tight_layout()                                  # Úprava rozložení (aby se popisky nepřekrývaly)
    plt.show()                                          # Zobrazení okna s grafem


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
    # Vizualizace blind search v 2D na 3D povrchu funkce.

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
    import matplotlib.pyplot as plt                        # Matplotlib pro vykreslení

    if bounds_2d is None:                                  # Pokud nejsou zadané hranice, tak default
        bounds_2d = get_default_bounds("sphere", 2)        

    # Předpočítáme mřížku a hodnoty funkce, aby byl povrch statický (jen body se mění)
    X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points=num_points)

    fig = plt.figure(figsize=(8, 6))                        # Nové okno (figure)
    ax = fig.add_subplot(111, projection='3d')              # 3D osa
    surf = ax.plot_surface(                                 # Vykreslení povrchu
        X,
        Y,
        Z,
        cmap='jet',
        edgecolor='k',
        linewidth=0.2,
        antialiased=True,
        rstride=1,
        cstride=1,
        alpha=0.35
    )
    ax.set_xlabel('x1')                                     # Popisek x
    ax.set_ylabel('x2')                                     # Popisek y
    ax.set_zlabel('f(x)')                                   # Popisek z
    ax.set_title('Vizualizace blind search (2D → 3D)')      # Titulek grafu

    (x_min, x_max), (y_min, y_max) = bounds_2d              # Rozbalení hranic pro pohodlí

    rng = random.Random(seed)                               # Deterministický random (pokud seed)

    def random_point_2d():                                  # Funkce vrátí náhodný bod (x,y) v mezích
        x = rng.uniform(x_min, x_max)                       # Náhodné x
        y = rng.uniform(y_min, y_max)                       # Náhodné y

        return x, y

    # Počáteční (náhodný) nejlepší bod
    best_x, best_y = random_point_2d()                      # Start pozice
    best_f = objective([best_x, best_y])                    # Jeho hodnota funkce

    # Předpřipravené grafické objekty (artists) pro efektivní aktualizaci
    candidates_scatter = ax.scatter([], [], [],             # Scatter kandidátů (prázdný na začátku)
                                     s=10, c='k', alpha=0.6, depthshade=False)
    best_scatter = ax.scatter([best_x], [best_y], [best_f],  # Scatter nejlepšího bodu
                               s=50, c='red', depthshade=False)

    plt.tight_layout()                                      
    plt.pause(0.3)                                          # Malá pauza pro inicializaci GUI

    g = 0                                                   # Počítadlo generací
    while g < g_max:                                        # Hlavní smyčka přes generace
        cand_x = []                                         # Seznam x kandidátů
        cand_y = []                                         # Seznam y kandidátů
        cand_z = []                                         # Seznam f(x) hodnot
        for _ in range(npop):                               # Vygeneruj populaci
            x, y = random_point_2d()                        # Náhodný bod
            z = objective([x, y])                           # Vyhodnocení funkce
            cand_x.append(x)                                # Ulož x
            cand_y.append(y)                                # Ulož y
            cand_z.append(z)                                # Ulož f(x)

        candidates_scatter._offsets3d = (cand_x, cand_y, cand_z)  # Aktualizace scatteru kandidátů

        best_idx = 0                                        # Index nejlepšího kandidáta v generaci
        best_gen_f = cand_z[0]                              # Hodnota nejlepšího kandidáta
        for i in range(1, len(cand_z)):                     # Projdeme ostatní kandidáty
            if cand_z[i] < best_gen_f:                      # Lepší nalezen
                best_gen_f = cand_z[i]                      # Aktualizace nejlepší hodnoty
                best_idx = i                                # Uložení indexu

        if best_gen_f < best_f:                             # Zlepšení globálního minima?
            best_x = cand_x[best_idx]                       # Aktualizace nejlepšího x
            best_y = cand_y[best_idx]                       # Aktualizace nejlepšího y
            best_f = best_gen_f                             # Aktualizace nejlepší hodnoty
            best_scatter._offsets3d = ([best_x],            # Posun markeru nejlepšího bodu
                                       [best_y],
                                       [best_f])

        plt.pause(pause_seconds)                            # Pauza pro animaci
        g += 1                                              # Další generace

    best_scatter.remove()                                   # Odstraníme původní marker nejlepšího bodu
    best_scatter = ax.scatter([best_x], [best_y], [best_f], # Vykreslíme zvýrazněný finální bod
                               s=120, c='lime', edgecolor='k', linewidth=0.6, depthshade=False)
    msg = f"KONEC: nejlepší bod = ({best_x:.4f}, {best_y:.4f}), f = {best_f:.6g}"  # Textová zpráva
    
    print(msg)                                              # Výpis do konzole
    
    ax.text2D(0.02, 0.95, msg,                              # Text také do grafu (2D souřadnice v ploše figury)
              transform=ax.transAxes,
              fontsize=9,
              color='lime',
              bbox=dict(facecolor='black', alpha=0.3, pad=4))
    
    plt.draw()                                              # Překreslení obrázku
    plt.show()                                              # Zobrazení finální podoby

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
    r"""
    Standardní (vícerozměrná) Levy funkce.

    Definice:
        Nejprve se provede transformace
            w_i = 1 + (x_i - 1) / 4

        f(x) = sin^2(π w_1)
               + Σ_{i=1..n-1} (w_i - 1)^2 * [ 1 + 10 * sin^2(π w_i + 1) ]
               + (w_n - 1)^2 * [ 1 + sin^2(2π w_n) ]

    Vlastnosti:
        - Doména obvykle x_i ∈ [-10, 10]
        - Globální minimum: x* = (1, ..., 1) → f(x*) = 0

    Parametry
    ---------
    params : list[float]
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota Levy funkce pro zadané `params`.
    """
    n = len(params)
    if n == 0:
        return 0.0

    # Transformace w_i
    w = []
    for x in params:
        w.append(1.0 + (x - 1.0) / 4.0)

    # První člen
    total = math.sin(math.pi * w[0]) ** 2

    # Prostřední součet (i = 1..n-1 => indexy 0..n-2)
    for i in range(0, n - 1):
        wi = w[i]
        total += (wi - 1.0) * (wi - 1.0) * (1.0 + 10.0 * (math.sin(math.pi * wi + 1.0) ** 2))

    # Poslední člen
    wn = w[-1]
    total += (wn - 1.0) * (wn - 1.0) * (1.0 + (math.sin(2.0 * math.pi * wn) ** 2))

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
    Blind saerch pro minimalizační úlohu.

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


def hill_climbing(
    objective: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    max_iter: int,
    step_sigma: float = 0.1,
    neighbours: int = 20,
    seed: Optional[int] = None,
    early_stop_no_improve: int = 50,
    visualize: bool = False,
    pause_seconds: float = 0.5,
    num_points: int = 160,
    surface_alpha: float = 1.0,
) -> Tuple[List[float], float]:
    """
    Hill Climbing (minimalizace) + VOLITELNÁ vizualizace pro 2D.

    Chování:
      - Samotný algoritmus proběhne celý předem.
      - Uložíme pouze kroky, kdy došlo ke ZLEPŠENÍ (seznam `path`).
      - Až poté spustíme animaci: zelený konečný bod je vidět od začátku
        a červený bod prochází postupně uložené kroky.

    Dodatečné parametry:
      visualize : bool      Má-li se (u 2D) zobrazit animace po doběhu algoritmu.
      pause_seconds : float Pauza mezi snímky animace.
      num_points : int      Hustota mřížky pro povrch při vykreslení.
            surface_alpha : float Průhlednost povrchu (1.0 = neprůhledný, např. 0.35 = částečně průsvitný).
    """
    import random
    rng = random.Random(seed)

    dim = len(bounds)

    # Pomocná funkce: vygeneruje jeden náhodný vektor v mezích
    def random_vector():
        v = []
        for low, high in bounds:
            value = rng.uniform(low, high) 
            if value < low:
                value = low
            if value > high:
                value = high
            v.append(value)
        return v

    # 1) Náhodný start a základní proměnné 
    current = random_vector()
    current_f = objective(current)
    best_x = list(current)
    best_f = float(current_f)

    # Uložíme první stav (start) do path (list zlepšení)
    path: List[Tuple[List[float], float]] = [(list(current), current_f)]

    it = 0
    no_improve = 0
    while it < max_iter:
        improved = False
        # 2) Generování sousedů
        for _ in range(neighbours):
            neighbor = []
            for i in range(dim):
                low, high = bounds[i]
                step = rng.gauss(0.0, step_sigma)
                cand = current[i] + step
                if cand < low:
                    cand = low
                if cand > high:
                    cand = high
                neighbor.append(cand)

            f_nei = objective(neighbor)
            if f_nei < current_f:  # našli jsme zlepšení
                current = neighbor
                current_f = f_nei
                improved = True
                if current_f < best_f:
                    best_x = list(current)
                    best_f = float(current_f)
                path.append((list(current), current_f))  # uložíme zlepšení

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        if early_stop_no_improve > 0 and no_improve >= early_stop_no_improve:
            break

        it += 1

    # 3) Vizualizace (jen pokud 2D a visualize=True)
    if visualize and dim == 2:
        import matplotlib.pyplot as plt

        # Vytvoříme povrch: použijeme pouze první dvě souřadnice
        X, Y, Z = evaluate_surface_2d(objective, bounds, num_points=num_points)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # Povrch – můžeme nastavit průhlednost (alpha) pro lepší viditelnost bodů
        alpha_val = surface_alpha
        if alpha_val < 0.0:
            alpha_val = 0.0
        if alpha_val > 1.0:
            alpha_val = 1.0
        ax.plot_surface(
            X, Y, Z,
            cmap='jet', edgecolor='k', linewidth=0.2, antialiased=True,
            rstride=1, cstride=1, alpha=alpha_val,
        )
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x)')
        ax.set_title('Hill Climbing – animace (předpočítaná cesta)')

        # Zelený finální bod (statický, viditelný hned od začátku)
        ax.scatter([best_x[0]], [best_x[1]], [best_f], s=140, c='lime', edgecolor='k', linewidth=0.6, depthshade=False)

        # Červený pohyblivý bod (začíná v prvním stavu path)
        first_state = path[0]
        moving = ax.scatter([first_state[0][0]], [first_state[0][1]], [first_state[1]], s=60, c='red', depthshade=False)
        plt.tight_layout()
        plt.pause(0.6)  # chvilka na prohlédnutí startu

        # Animace přes uložené zlepšovací kroky
        for state, value in path[1:]:
            moving._offsets3d = ([state[0]], [state[1]], [value])
            plt.pause(pause_seconds)

        msg = f"HILL CLIMBING HOTOVO: best=({best_x[0]:.4f}, {best_x[1]:.4f}), f={best_f:.6g}, kroku={len(path)}"
        print(msg)
        ax.text2D(0.02, 0.94, msg, transform=ax.transAxes, fontsize=9, color='lime',
                  bbox=dict(facecolor='black', alpha=0.3, pad=4))
        plt.show()

    return best_x, best_f