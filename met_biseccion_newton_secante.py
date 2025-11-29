import numpy as np
import math
import matplotlib.pyplot as plt
# los metodos son generales, solo se tiene que cambian
#la "definicion de la funcion" que se encuentra abajo para 

#BISECCIÓN
def biseccion(a, b, tol):
    print("\n MÉTODO DE BISECCIÓN")
    print("Iter |        a       |        b       |        x       |     f(x)")

    it = 0
    while abs(b - a) > tol:
        x = (a + b) / 2
        print(f"{it:4d} | {a:12.8f} | {b:12.8f} | {x:12.8f} | {f(x):12.4e}")

        if f(a) * f(x) < 0:
            b = x
        else:
            a = x
        it += 1

    print("\nRaíz aproximada (Bisección):", x)
    return x


#NEWTON
def newton(x0, tol, max_iter=50):
    print("\n MÉTODO DE NEWTON")
    print("Iter |        x        |       f(x)       |      f'(x)")

    for i in range(max_iter):
        fx = f(x0)
        dfx = df(x0)

        print(f"{i:4d} | {x0:12.8f} | {fx:14.4e} | {dfx:14.4e}")

        if abs(fx) < tol:
            print("\nRaíz aproximada (Newton):", x0)
            return x0
        
        x0 = x0 - fx/dfx

    print("Newton no converge")
    return None


# SECANTE
def secante(x0, x1, tol, max_iter=50):
    print("\n MÉTODO DE LA SECANTE ")
    print("Iter |        x_n      |      f(x_n)")

    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)

        print(f"{i:4d} | {x1:12.8f} | {fx1:12.4e}")

        if abs(fx1) < tol:
            print("\nRaíz aproximada (Secante):", x1)
            return x1

        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        x0, x1 = x1, x2

    print("Secante no converge")
    return None



#GRAFICA

def graficar_funcion(func, xmin, xmax, roots=None, puntos=500):
    X = np.linspace(xmin, xmax, puntos)
    Y = [func(x) for x in X]

    plt.figure(figsize=(9,5))
    plt.axhline(0, color='black', linewidth=1)
    plt.plot(X, Y, label="f(x)")

    if roots:
        for r in roots:
            plt.plot(r, func(r), 'ro', markersize=8, label=f"Raíz ≈ {r:.5f}")

    plt.title("Gráfica de la función f(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.show()

#DEFINIR DE LA FUNCIÓN

def f(x):
    """Función a evaluar"""
    return x**3 - x**2 * math.exp(-0.5*x) - 3*x + 1

def df(x):
    """Derivada de f(x)"""
    return (3*x**2
            - (2*x * math.exp(-0.5*x) + x**2 * (-0.5) * math.exp(-0.5*x))
            - 3)

print(" Resolviendo la función:")
print("   f(x)   = x^3 - x^2 * e^{-0.5x} - 3x + 1")
print("   f'(x)  = 3x^2 - [2x e^{-0.5x} + x^2(-0.5)e^{-0.5x}] - 3")

 
#EJECUCION EJEMPLO: PROBLEMA 3 raiz 2

tol = 0.0005

r_bis = biseccion(0.0, 0.5, tol)
r_new = newton(0.0, tol)
r_sec = secante(0.0, 0.5, tol)

# Graficar función con raíces encontradas
graficar_funcion(f, xmin=-3, xmax=4, roots=[r_bis, r_new, r_sec])
