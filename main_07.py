import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.special
from scipy.optimize import curve_fit
import matplotlib.colors
from matplotlib.collections import LineCollection


def vec_euler(f, x0, t):
    """
    Takes a function f with a multivalued output, a vector of initial values x0 and solves IVP using Euler's method
    """
    h = t[1] - t[0]
    x = x0
    output = []
    for time in t:
        row = []
        f_vec = f(x, time)
        n = np.shape(f_vec)[0]  # Number of function returns
        for i in range(n):
            x[i] += h*f_vec[i]
            row.append(x[i])
        output.append(row)

    return np.column_stack(np.array(output))


def vec_midpoint(f, x0, t):
    h = t[1] - t[0]
    x = x0
    output = []
    for time in t:
        row = []
        f_vec = f(x, time)
        n = np.shape(f_vec)[0]
        k1 = [h*f_vec[i]/2 for i in range(n)]
        f_vec2 = f(x + k1, time + h/2)
        for i in range(n):
            x[i] += h * f_vec2[i]
            row.append(x[i])

        output.append(row)

    return np.column_stack(np.array(output))


def vec_rk4(f, x0, t):
    h = t[1] - t[0]
    x = x0
    output = []
    for time in t:
        k1 = h * np.asarray(f(x, time))
        k2 = h * np.asarray(f(x + 1/2 * k1, time + h/2))
        k3 = h * np.asarray(f(x + 1/2 * k2, time + h/2))
        k4 = h * np.asarray(f(x + k3, time + h))
        x += 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        output.append(x)

    return np.column_stack(output)


def symp_verlet(f, x0, v0, t):
    """
    For solving 2. order IVP where v = dx/dt using Verlet's algorithm
    """
    h = t[1] - t[0]
    x = x0
    v = v0
    output = []
    for time in t:
        first_eval = f(x, time)
        x += h*v + 0.5*h**2 * first_eval
        second_eval = f(x, time)
        v += 0.5*h*(first_eval + second_eval)
        output.append([x, v])

    return np.column_stack(np.array(output))


def symp_pefrl(f, x0, v0, t):
    """
    Solves 2. order IVP using "Position Extended Forest-Ruth Like" algorithm of Omelyan et al.

    INPUTS:
    f: function to equal dx^2/dt^2
    x0: initial value for f(t[0])
    v0: initial value for df/dt(v[0])
    t: array of times to evaluate function

    OUTPUT:
    output: 2D Array of evaluated f(x, t) values and evaluated df/dt(x, t) values
    """
    xi = 0.1786178958448091
    lam = -0.2123418310626054
    chi = -0.6626458266981849e-1
    # h = t[1] - t[0]
    t_prev = t[0]
    x = x0
    v = v0
    output = []
    for time in t:
        h = time - t_prev
        x += xi*h*v
        v += (1 - 2*lam)*0.5*h*f(x, time)
        x += chi * h * v
        v += lam*h*f(x, time)
        x += (1 - 2*(chi + xi))*h*v
        v += lam*h*f(x, time)
        x += chi*h*v
        v += (1 - 2*lam)*0.5*h*f(x, time)
        x += xi*h*v
        t_prev = time
        output.append([x, v])

    return np.column_stack(np.array(output))


def symp_leapfrog(f, x0, v0, t):
    """
    Solves 2. order IVP where v = dx/dt using standard leapfrog integrator
    INPUTS:
    f: function to equal dx^2/dt^2
    x0: initial value for f(t[0])
    v0: initial value for df/dt(v[0])
    t: array of times to evaluate function
    OUTPUT:
    output: 2D Array of evaluated f(x, t) values and evaluated df/dt(x, t) values
    """
    h = t[1] - t[0]
    x = x0
    v = v0
    output = []
    for time in t:
        v_half = v + f(x, time) * h/2
        x += v_half*h
        v = v_half + f(x, time) * h/2
        output.append([x, v])

    return np.column_stack(np.array(output))


def symp_yoshida(f, x0, v0, t):
    """
    Solves 2. order IVP where v = dx/dt using higher order leapfrog integrator by Haruo Yoshida

    INPUTS:
    f: function to equal dx^2/dt^2
    x0: initial value for f(t[0])
    v0: initial value for df/dt(v[0])
    t: array of times to evaluate function

    OUTPUT:
    output: 2D Array of evaluated f(x, t) values and evaluated df/dt(x, t) values
    """
    c1 = 0.6756035959798288170238
    c2 = -0.1756035959798288170235
    c3 = -0.1756035959798288170235
    c4 = 0.6756035959798288170238
    d1 = 1.351207191959657634048
    d2 = -1.702414383919315268095
    d3 = 1.351207191959657634048
    h = t[1] - t[0]
    x = x0
    v = v0
    output = []
    for time in t:
        x1 = x + c1 * v * h
        v1 = v + d1 * f(x1, time) * h
        x2 = x1 + c2 * v1 * h
        v2 = v1 + d2 * f(x2, time) * h
        x3 = x2 + c3 * v2 * h
        v3 = v2 + d3 * f(x3, time) * h
        x = x3 + c4 * v3 * h
        v = v3
        output.append([x, v])

    return np.column_stack(np.array(output))


def dpdt(x, t):  # dp/dt = -sin(x)
    return -np.sin(x[0])


def newton(x, t):  # d^2x/dt^2  = -sin(x)
    return -np.sin(x)


def function(x, t):
    return x[1], -np.sin(x[0])


def energija(x, v, w0):
    return 1 - np.cos(x) + v**2/(0.5 * w0**2)


def exact(t, x0, w0):
    return 2*np.arcsin(np.sin(x0/2) *
                       scipy.special.ellipj(scipy.special.ellipk((np.sin(x0/2))**2) - w0*t, (np.sin(x0/2)**2))[0])


def square(x, a, b, c):
    return a*x**2 + b*x + c


def step_avg_error(a, b, method, f, x0, exact, time_start, time_stop, *args):
    """
    Returns average errors of method at different step values.

    INPUTS:
    a: n range start
    b: n range stop
    method: callable function for method to solve ODE
    f: callable function equaling x' = f(x,t) for method
    x0: starting parameter for x(t[0])
    exact: callable function for exact values to calculate error
    time_start: time range start
    time_stop: time range stop

    OUTPUT:
    k: array of step sizes
    output: array of corresponding average errors
    """
    output = []
    n = np.arange(a, b)
    k = []
    for item in n:  # Item is n division for times
        print(item)
        t = np.linspace(time_start, time_stop, item)
        f_method = method(f, x0, t, *args)
        f_exact = exact(t, x_0, w_0)
        output.append(np.median(np.abs(f_method - f_exact)))
        k.append((time_stop-time_start)/item)

    return np.array(k), np.array(output)


def symp_tep_avg_error(a, b, method, f, x0, v0, exact, time_start, time_stop):
    """
    Returns average errors of method at different step values.

    INPUTS:
    a: n range start
    b: n range stop
    method: callable function for method to solve ODE
    f: callable function equaling x' = f(x,t) for method
    x0: starting parameter for x(t[0])
    exact: callable function for exact values to calculate error
    time_start: time range start
    time_stop: time range stop

    OUTPUT:
    k: array of step sizes
    output: array of corresponding average errors
    """
    output = []
    n = np.arange(a, b)
    k = []
    for item in n:  # Item is n division for times
        print(item)
        t = np.linspace(time_start, time_stop, item)
        f_method = method(f, x0, v0, t)
        f_exact = exact(t, x_0, w_0)
        output.append(np.median(np.abs(f_method - f_exact)))
        k.append((time_stop-time_start)/item)

    return np.array(k), np.array(output)

# Parameters
x_0 = 1
p_0 = 0
g = 9.81  # m/s^2
l = 4  # m
# w_0 = np.sqrt(g / l)
w_0 = 1
# Interval parameters
a, b = (0.0, 20.0)
n = 1000
t_int = np.linspace(a, b, n)

# Data
exact_solve = exact(t_int, x_0, w_0)
sol = vec_euler(function, [x_0, p_0], t_int)
sol1 = symp_verlet(newton, x_0, p_0, t_int)
sol2 = vec_midpoint(function, [x_0, p_0], t_int)
sol3 = symp_pefrl(newton, x_0, p_0, t_int)
sol4 = symp_yoshida(newton, x_0, p_0, t_int)
sol5 = symp_leapfrog(newton, x_0, p_0, t_int)
sol6 = vec_rk4(function, [x_0, p_0], t_int)
data = np.column_stack(scipy.integrate.odeint(function, [p_0, x_0], t_int))

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))
# ax1.plot(t_int, exact_solve, label="Exact", c="#AF1B3F")
# # ax1.plot(t_int, sol1[0], label="Verlet", c="#FCFFA2")
# # ax1.plot(t_int, sol3[0], label="PEFRL", c="#B5FFA7")
# # ax1.plot(t_int, sol3[0], label="Yoshida", c="#85F5FF")
# # ax1.plot(t_int, sol4[0], label="Leapfrog", c="#89B7FF")
# ax1.plot(t_int, data[1], label="Scipy", c="#FFCC8E")
# ax1.plot(t_int, sol2[0], label="Midpoint", c="#AC9DFF")
# ax1.plot(t_int, sol[0], label="Euler", c="#FFB0FF")

# ax1.set_xlim(0, 20)
# ax1.set_title("Rešitev Eulerjeve metode pri koraku {}".format((b-a)/n))
# ax1.set_title("Rešitev nesimplektičnih metod pri koraku {}".format((b-a)/n))
# ax1.set_xlabel("t [s]")
# ax1.set_ylabel("Amplituda")
# ax1.legend()

# ax2.plot(t_int, np.abs(np.abs(exact_solve - data[1])/exact_solve), label="Scipy", c="#FFCC8E")
# ax2.plot(t_int, np.abs(np.abs(exact_solve - sol1[0])/exact_solve), label="Verlet", c="#FCFFA2")
# ax2.plot(t_int, np.abs(np.abs(exact_solve - sol3[0])/exact_solve), label="PEFRL", c="#B5FFA7")
# ax2.plot(t_int, np.abs(np.abs(exact_solve - sol4[0])/exact_solve), label="Yoshida", c="#85F5FF")
# ax2.plot(t_int, np.abs(np.abs(exact_solve - sol5[0])/exact_solve), label="Leapfrog", c="#89B7FF")
# ax2.plot(t_int, np.abs(np.abs(exact_solve - sol2[0])/exact_solve), label="Midpoint", c="#AC9DFF")
# ax2.plot(t_int, np.abs(np.abs(exact_solve - sol[0])/exact_solve), label="Euler", c="#FFB0FF")
# ax2.set_title("Relativna napaka")
# ax2.set_yscale("log")
# ax2.set_xscale("log")
# ax2.set_xlabel("t [s]")
# ax2.set_ylabel(r"$\frac{|exact - data|}{exact}$")
# ax2.legend()


# ABS NAPAKA EULER
# ax2.plot(t_int, np.abs(exact_solve - sol[0]), label="Euler", c="#FFB0FF")
# fitpar, fitcov = curve_fit(square, xdata=t_int, ydata=np.abs(exact_solve - sol[0]))
# yfit = square(t_int, fitpar[0], fitpar[1], fitpar[2])
# fittext= "Quadratic fit: $y = ax^2 + bx+ c$\na = {} ± {}\nb = {} ± {}\nc = {} ± {}".format(format(fitpar[0], ".4e"), format(fitcov[0][0]**0.5, ".4e"),
#                                                                                            format(fitpar[1], ".4e"), format(fitcov[1][1]**0.5, ".4e"),
#                                                                                            format(fitpar[2], ".4e"), format(fitcov[2][2]**0.5, ".4e"))
# ax2.text(0.5, 0.4, fittext, ha="left", va="center", size=10, transform=ax2.transAxes, bbox=dict(facecolor="#a9f5ee", alpha=0.5))
# ax2.plot(t_int, yfit, label="Fit", c="#8CA0D7")
# ax2.set_title("Absolutna napaka")
# ax2.set_yscale("log")
# # ax2.set_xscale("log")
# ax2.set_xlabel("t [s]")
# ax2.set_ylabel(r"$|exact - data|$")
# ax2.legend()

# ax3.plot(data[1], data[0], label="Scipy", c="#FFCC8E")
# ax3.plot(sol1[1], sol1[0], label="Verlet", c="#FCFFA2")
# ax3.plot(sol3[1], sol3[0], label="PEFRL", c="#B5FFA7")
# ax3.plot(sol4[1], sol4[0], label="Yoshida", c="#85F5FF")
# ax3.plot(sol5[1], sol5[0], label="Leapfrog", c="#89B7FF")
# ax3.plot(sol2[0], sol2[1], label="Midpoint", c="#AC9DFF")
# ax3.plot(sol[0], sol[1], label="Euler", c="#FFB0FF")
# ax3.set_title("Fazni portret")
# ax3.set_xlabel("x")
# ax3.set_ylabel("v")
# ax3.legend()
#
# fig.subplots_adjust(top=0.94, bottom=0.06, hspace=0.35, left=0.14, right=0.94)
# plt.show()


# Required step size plot
# fig, ax = plt.subplots()
# stepdata = np.arange(100, 2000)
# out = []
# for j in stepdata:
#     time = np.linspace(a, b, j)
#     solve = vec_euler(function, [x_0, p_0], time)
#     exact_solve = exact(time, x_0, w_0)
#     # plt.plot(time, solve[0])
#     out.append(np.column_stack([time, solve[0]]))
#
# out = np.asarray(out)
# segments = out
# num_colors = solve.shape[0]
# print(segments)
# line_segments = LineCollection(segments, cmap="plasma")
# step = (b - a)/stepdata
# line_segments.set_array(step)
# ax.add_collection(line_segments)
# plt.title("Kvarjenje rešitve pri Eulerjevi metodi")
# plt.xlabel("t [s]")
# plt.ylabel(r"Amplituda")
# plt.xlim(0, 20)
# plt.ylim(-10, 10)
# plt.colorbar(mappable=line_segments, label="Velikost koraka h")
# plt.show()




# ===============================================================

# Final solution
# fig, ax = plt.subplots()
# solution = symp_verlet(newton, x_0, p_0, t_int)
# plt.plot(t_int, solution[0], c="#7F68FF")
# plt.title("Rešitev matematičnega nihala")
# plt.xlabel("t [s]")
# plt.ylabel(r"Amplituda")
# plt.show()
#
# fig, ax = plt.subplots()
# plt.plot(solution[0], solution[1], c="#7F68FF")
# plt.title("Fazni portret matematičnega nihala")
# plt.xlabel("x")
# plt.ylabel(r"v")
# plt.show()
#

# Energy plots
energy_euler = energija(sol[0], sol[1], w_0)
energy_verlet = energija(sol1[0], sol[1], w_0)
energy_midpoint = energija(sol2[0], sol2[1], w_0)
energy_pefrl = energija(sol3[0], sol[1], w_0)
plt.plot(t_int, energy_euler)
plt.plot(t_int, energy_verlet)
plt.plot(t_int, energy_midpoint)
plt.plot(t_int, energy_pefrl)
plt.show()


