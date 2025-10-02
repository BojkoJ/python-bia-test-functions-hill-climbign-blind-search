# BIA – Cvičení 1 a 2 (Biologicky inspirované algoritmy)

Tento repozitář obsahuje řešení 1. a 2. cvičení z předmětu Biologicky inspirované algoritmy.
Cílem je mít implementace testovacích funkcí pro optimalizaci a základní „blind search“ (slepé vyhledávání).

https://michaelmachu.eu/data/pdf/bia/Exercise1.pdf

https://michaelmachu.eu/data/pdf/bia/Exercise2.pdf

## Obsah

-   `main.py` – modulové implementace funkcí a jednoduchý algoritmus blind search:
    -   Testovací funkce (minimizační): Sphere, Schwefel, Rosenbrock, Rastrigin, Griewank, Levy (základní), Michalewicz, Zakharov, Ackley
    -   Algoritmus: `blind_search(objective, bounds, npop, g_max, seed=None)`
    -   Algoritmus: `def hill_climbing(objective, bounds, max_iter, step_sigma, neighbours, seed, early_stop_no_improve, visualize, pause_seconds, num_points, surface_alpha)`
-   `tests/` – testy (každá funkce má vlastní soubor `*_test.py`).

## Jak spustit testy (Windows / PowerShell)

Testy jsou nezávislé a spouští se přímo jednotlivé soubory:

```powershell
# Sphere
python "tests/sphere_test.py"

# Schwefel
python "tests/schwefel_test.py"

# Rosenbrock
python "tests/rosenbrock_test.py"

# Rastrigin
python "tests/rastrigin_test.py"

# Griewank
python "tests/griewank_test.py"

# Levy (základní varianta)
python "tests/levy_test.py"

# Michalewicz
python "tests/michalewicz_test.py"

# Zakharov
python "tests/zakharov_test.py"

# Ackley
python "tests/ackley_test.py"

# Blind Search (na Sphere)
python "tests/blind_search_test.py"

# Hill Climbing (na Sphere)
python "tests/hill_climbing_test.py"
```

## Blind search – stručně

-   Algoritmus dostane cílovou funkci `objective(x)` a meze `bounds`.
-   V každé generaci náhodně vygeneruje NP kandidátů uvnitř mezí, vybere nejlepší a případně jím nahradí dosavadní nejlepší řešení.
-   Po `g_max` generacích vrátí nejlepší nalezený vektor `best_x` a jeho hodnotu `best_f`.

## Hill Climbing - stručně

-    Funkce algoritmu může provádět i vizualizaci po předání visualize
-    V každé iteraci vygeneruje sousedy, každého souseda posune pomocí náhodně generovaného stepu
-    Pak porovná sousedy, zvolí nejlepšího (nejmenšího - protože minimalizační funkce) pro danou populaci. Uloží do path a pokračuje.

Příklad (2D Sphere):

```python
from main import blind_search, sphere
bounds = [(-5.12, 5.12)] * 2
best_x, best_f = blind_search(sphere, bounds, npop=50, g_max=50, seed=123)
print(best_x, best_f)
```

## Poznámky

-   Všechny implementované funkce jsou chápány jako minimalizační (menší je lepší). Pro maximalizaci lze předat `objective=lambda x: -g(x)`.
