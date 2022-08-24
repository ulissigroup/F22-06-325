---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{margin} Adaptation!
This lecture was also adapted from lecture notes in John Kitchin's excellent 06-623 course. His lecture notes are included in the helpful resources link if you want to know more details about how numerical methods work. These notes also contain content from Prof. Ulissi's 06-262 course. 
```

+++

# Local Optimization and Curve Fitting


`````{note}
This lecture is going to:
* Quickly review why we care about finding minima and maxima of functions
* Demonstrate two methods for finding minima/maxima:
    * Using a non-linear solver to find where the gradient is zero
    * Using a local optimizer from scipy
* Show that finding minima and maxima are the same problem

After covering these ideas, we will practice on a problem that we can also solve with ODE events!

`````

+++ {"id": "36LXSXgpdsge"}

## Function extrema

+++ {"id": "WnZv-Bgidsgf"}

It is pretty common to need to find extreme values of a function in engineering analysis. An extreme value is often a maximum or minimum in a function, and we seek them when we want to maximize a profit function, or minimize a cost function, identify a maximum safe operating condition, etc.

Let's consider an example function with a graphical solution approach. We want a quantitative estimate of the minimum in this function.

```{code-cell} ipython3
:id: Q9UwDQl3dsgf
:outputId: d0e8e0b0-7361-4976-eb11-1690733ad122

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2 + np.exp(-5 * x**2)


x = np.linspace(0, 2)
y = f(x)
plt.plot(x, y)
```

+++ {"id": "ff6Huxiidsgh"}

You can see there is a minimum near 0.6. We can find the minimum in a crude kind of way by finding the index of the minimum value in the y-array, and then getting the corresponding value of the x-array. You control the accuracy of this answer by the number of points you discretize the function over.

```{code-cell} ipython3
:id: KJ1CL2uXdsgh
:outputId: 043346d6-e510-4fd0-c2a3-0c99a3d094b6

x = np.linspace(0, 2, 50)
y = f(x)
i = np.argmin(y)
x[i]
```

+++ {"id": "ss9FIkxcdsgi"}

`````{tip}

**Pros/cons of finding minima/maxima by inspection!**

Pros:
1.  It is easy.
2.  You *see* the whole domain you are looking at, and it is easy to see how many extrema their are

Cons:
1.  *Lot's* of function evaluations. Imagine if it took a long time to compute each value.
2.  Somewhat tedious.
3.  Not so easy to reproduce
4.  Not scalable to large problems, your time to do this becomes a limiting factor.
`````

+++ {"id": "Y-WUmY0Odsgi"}

### Find the derivative, and solve for where it is zero

+++ {"id": "0KYh-8qkdsgi"}

We can also derive the first derivative:

$y' = 2x + e^{-5 x^2} (-10x)$

and solve it for zero using fsolve.

```{code-cell} ipython3
:id: t8pSMifjdsgj
:outputId: 62938e95-52b1-4fb3-dbe3-38f6fbf06536

def yp(x):
    return 2 * x + np.exp(-5 * x**2) * (-10 * x)


from scipy.optimize import fsolve

fsolve(yp, 0.5)
```

+++ {"id": "yPYMTc2Ddsgj"}

These two answer agree to 5 decimal places.

This depends on your ability to correctly derive and implement the derivative. It is good to know you can solve this problem by more than one method. Here, we use a numerical derivative in the function instead to check our derivative. You can check the convergence of the derivative by varying the dx.

```{code-cell} ipython3
:id: 0VuIZkAQdsgj
:outputId: dc8022de-743b-4d63-e6cd-21ee8eb6f5f1

from scipy.misc import derivative


def ypd(x):
    return derivative(f, x, dx=1e-6)


fsolve(ypd, 0.5)
```

+++ {"id": "ol6Iafxvdsgk"}

These look the same within tolerance. This is not a beautiful solution, but it is hard to argue with success here!

`````{tip}

**Pros/cons of finding minima/maxima by root finding of the derivatives!**

Pros:
1.  We've turned a new problem into a problem we already know how to solve.

Cons:
1.  You have to do a derivative by hand or use a numerical estimate.
2.  You get minima, maxima, and saddle points.
`````

+++ {"id": "tM3Sub3Cdsgl"}

### Standard python approach: scipy.optimize.minimize

+++ {"id": "gVbU1KXjdsgl"}

Here is the basic use of scipy.optimize.minimize. As always, we should plot the answer where feasible to make sure it is the minimum we wanted.

`````{seealso}
Full documentation and notes on types of algorithms: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
`````

```{code-cell} ipython3
:id: mJDxo14Adsgm
:outputId: a1121bee-cb03-41f3-e55c-dadb66c1549a

from scipy.optimize import minimize


def f(x):
    return x**2 + np.exp(-5 * x**2)


guess = 0.5
sol = minimize(f, guess)
sol
```

```{code-cell} ipython3
:id: YDYDIpQUdsgm
:outputId: 01c5369d-fbcb-4f31-dcb3-a5099ec1a7ca

x = np.linspace(0, 2)
y = f(x)

plt.plot(x, y, "b-")
plt.plot(sol.x, f(sol.x), "ro")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["f(x)", "fmin"])
```

+++ {"id": "25UDLJxtdsgm"}

Note this answer is only the same in the first 4 decimal places. Remember that these iterative approaches stop when a tolerance is met. Check the defaults on fmin!

+++ {"id": "vLlvw6_Jdsgm"}

### Multiple minima

+++ {"id": "n6LAslCrdsgm"}

It is possible for functions to have more than one minimum. In this case, your guess will determine which minimum is found. Here is an example where there is a minimum near 2.2, and one near 4.5.

```{code-cell} ipython3
:id: bt3oS9eydsgm
:outputId: 312c4fae-beea-4e29-cd81-ac88f78e5edd

def h(x):
    return 2 + np.cos(x) + np.cos(2 * x - 0.5) / 2


x = np.linspace(0, 2 * np.pi)

plt.plot(x, h(x))
plt.xlabel("x")
plt.ylabel("h(x)")
```

+++ {"id": "szidZcMHdsgm"}

This guess finds the one near 2.2:

```{code-cell} ipython3
:id: P1G_SCoCdsgm
:outputId: 863e0317-0a3b-4d2b-cb93-9121dbf12420

minimize(h, 2)
```

+++ {"id": "oemdtERFdsgn"}

and this guess finds the one near 4.5

```{code-cell} ipython3
:id: n_kVCMogdsgn
:outputId: c03629c7-342b-413c-86e3-fe2bf48334fb

minimize(h, 4)
```

+++ {"id": "iof-cVlKdsgn"}

You have to decide which one is better for the problem at hand. If this were a cost function, the one at the lower cost is probably better! Note that all we can say here is which one is lower in the interval we are looking at. By inspection of the function, you can see it will be periodic, so there will be many other minima that also exist.


`````{seealso}

Questions about local vs global minima and their use in decision making for engineering can be very interesting. You will discuss this in much more detail in your second mini (optimization) this semester!
`````

+++ {"id": "KDFDbY0Adsgn"}

### Finding maxima

+++ {"id": "fx2z_UFcdsgn"}

`minimize` is for finding *minima* (no surprise!). We can use it to find maxima though, by finding the *minima* of $-f(x)$. You can see here that when we plot $-h(x)$ the minima become maxima, and vice-versa. Now you can see there are two definite minima, one near zero, and one near 3.5, which correspond to the maxima of $h(x)$.

```{code-cell} ipython3
:id: 8dZFxsEidsgn
:outputId: 51d00a2a-7373-4f18-a8ef-1b875316cf83

plt.plot(x, -h(x))
plt.xlabel("x")
plt.ylabel("-h(x)")
```

+++ {"id": "1QXm-JUzdsgn"}

The standard way to use minimize is to define an optional argument for the sign that defaults to one. Then, when we call fmin, we will pass -1 as the sign to the function, so we find the minimum of -h(x). Then, we evaluate h(x) at that x-value to get the actual value of the maximum. It is not necessary do this, you can also manually pass around the sign and try to keep it straight.

Here is an example to find the maximum near 3.5.

```{code-cell} ipython3
:id: XlR0j1Drdsgn
:outputId: 3a96db79-cb3e-4d08-a198-8516940dbbea

def h(x, sign=1):
    return sign * (2 + np.cos(x) + np.cos(2 * x - 0.5) / 2)


sol = minimize(h, 3.5, args=(-1,))  # set sign=-1 here to minimize -h(x)
print(h(sol.x))  # sign defaults to 1 here, so we get the maximum value

plt.plot(x, h(x))
plt.plot(sol.x, h(sol.x), "ro")
plt.xlabel("x")
plt.ylabel("h(x)")
```

+++ {"id": "IHKDpXeedsgn"}

Once again, here you have to decide which maximum is relevant

+++ {"id": "gXyE36TCdsgn"}

### Example: Application to maximizing profit in a PFR

+++ {"id": "JAYePfYNdsgo"}

Compound X with concentration of $C_{X0} = 2.5$ kmol / m$^3$ at a flow rate of 12 m$^3$/min is converted to Y in a first order reaction with a rate constant of 30 1/min in a tubular reactor. The value of Y is $\$$1.5/kmol. The cost of operation is $\$$2.50 per minute per m$^3$. Find the reactor length that maximizes the profit (profit is value of products minus operating costs).

First, consider why there is a maximum. At low volumes the operating cost is low, and the production of Y is low. At high volumes, you maximize production of Y, so you have the most value, but the operating costs go up (to infinity for complete conversion!). Somewhere in the middle is where a maximum is.

Here are some relevant constants.

```{code-cell} ipython3
:id: H88WBDpUdsgo
:outputId: a7fd9f66-aadf-4900-bc4f-75e4df39d7f6

cost = 2.5  # dollar/min/m**3
y_value = 1.5  # dollar / mol

Cx0 = 2.5  # kmol / m**3
v0 = 12.0  # m**3 / min

k = 30.0  # 1/min
```

+++ {"id": "x_JosbU-dsgo"}

To compute the profit as a function of reactor volume, we need to compute how much Y is produced, then multiply that by the value of Y and subtract the operating cost. To compute how much Y is produced, we use a mole balance on X and Y, and integrate it to the volume to get the molar flows of X and Y. I like to write mole balances like this.

```{code-cell} ipython3
:id: 8NLNbeUqdsgo
:outputId: 70cc0c2a-8183-443e-cb93-74ef2b20208c

def dFdV(V, F):
    "PFR mole balances on X and Y."
    Fx, Fy = F
    Cx = Fx / v0
    rx = -k * Cx
    ry = -rx

    dFdX = rx
    dFdY = ry
    return [dFdX, dFdY]


F0 = [Cx0 * v0, 0.0]  # Fx0  # Fy0
```

+++ {"id": "v-7ZZ8eDdsgo"}

Now, we can write a profit function. It will take a V as the argument, integrate the PFR to that volume to find the molar exit flow rates, and then compute the profit.

```{code-cell} ipython3
:id: qlEnO4Vgdsgo
:outputId: ff04a2d2-d8c6-4905-f8ab-8a93185830e0

import numpy as np
from scipy.integrate import solve_ivp


def profit(V, sign=1):
    Vspan = (0, V)
    sol = solve_ivp(dFdV, Vspan, F0)
    Fx, Fy = sol.y
    Fy_exit = Fy[-1]
    return sign * (Fy_exit * y_value - cost * V)
```

+++ {"id": "Gc71GFovdsgo"}

It is always a good idea to plot the profit function. We use a list comprehension here because the profit function is not *vectorized*, which means we cannot pass an array of volumes in and get an array of profits out.

```{code-cell} ipython3
:id: QwHc7MRYdsgo
:outputId: 056c0d2a-e4af-4625-abd6-37da6dee0028

Vspan = np.linspace(0, 4)
profit_array = [profit(V) for V in Vspan]

import matplotlib.pyplot as plt

plt.plot(Vspan, profit_array)
plt.xlabel("V")
plt.ylabel("profit")
```

+++ {"id": "VRHIDvV9dsgo"}

You can see from this plot there is a maximum near V=1.5. We can use that as a guess for fmin.

```{code-cell} ipython3
:id: BbOkzYrhdsgo
:outputId: 501165f5-fb45-4464-8c9f-7dfbfd64d8a7

from scipy.optimize import fmin

sol = minimize(profit, 1.5, args=(-1,))

print(
    f"The optimal volume is {sol.x[0]:1.2f} m^3 with a profit of ${profit(sol.x[0]):1.2f}."
)
```

+++ {"id": "a3QJdbUxdsgo"}

This problem highlights the opportunities we have to integrate many ideas together to solve complex problems. We have integration of an ODE, nonlinear algebra/minimization, with graphical estimates of the solution.

+++

## **Practice:** Can you solve this with an event and solve\_ivp?

+++ {"id": "NuhAt7rYgyuQ"}

## Regression of data is a form of function minimization

+++ {"id": "sj_moG5VgyuR"}

When we say regression, we really mean find some parameters of a model that best reproduces some known data. By "best reproduces" we mean the sum of all the errors between the values predicted by the model, and the real data is minimized.

Suppose we have the following data that shows how the energy of a material depends on the volume of the material.

```{code-cell} ipython3
:id: Wnd2fDHDgyuR
:outputId: d90784b9-8191-4ccd-bab6-8cbd9e28c808

import matplotlib.pyplot as plt
import numpy as np

volumes = np.array([13.71, 14.82, 16.0, 17.23, 18.52])
energies = np.array([-56.29, -56.41, -56.46, -56.463, -56.41])

plt.plot(volumes, energies, "bo")
plt.xlabel("V")
plt.ylabel("E")
```

+++ {"id": "ujgkss0mgyuS"}

In Materials Science we often want to fit an equation of state to this data. We will use this equation:

$E = E_0 + \frac{B_0 V}{B_0'}\left(\frac{(V_0 / V)^{B_0'}}{B_0' - 1} + 1 \right) - \frac{V_0 B_0}{B_0' - 1}$

from [https://journals.aps.org/prb/pdf/10.1103/PhysRevB.28.5480](https://journals.aps.org/prb/pdf/10.1103/PhysRevB.28.5480). In this model there are four parameters:

| name|desc|
|---|---|
| E\_0|energy at the minimim|
| B\_0|bulk modulus|
| B\_0'|first derivative of the bulk modulus|
| V\_0|volume at the energy minimum|

We would like to find the value of these parameters that best fits the data above. That means, find the set of parameters that minimize the sum of the squared errors between the model and data.

First we need a function that will use the parameters and return the energy for a given volume.

```{code-cell} ipython3
:id: yFqa2eb7gyuT
:outputId: e49f2283-0fc7-4946-d5b8-3e472b3fbfd4

def Murnaghan(parameters, vol):
    "From PRB 28,5480 (1983)"
    E0, B0, BP, V0 = parameters
    E = E0 + B0 * vol / BP * (((V0 / vol) ** BP) / (BP - 1) + 1) - V0 * B0 / (BP - 1.0)

    return E
```

+++ {"id": "dvdHs4gQgyuT"}

Next, we need a function that computes the summed squared errors for a set of parameters. The use of squared errors is preferable in many cases to the absolute values because it has a continuous derivative. We will learn more about this later.

```{code-cell} ipython3
:id: YME4qTEsgyuT
:outputId: 49b355e7-0737-4d74-fa2c-c33b349e8d4b

def objective(pars):
    err = energies - Murnaghan(pars, volumes)
    return np.sum(err**2)  # we return the summed squared error directly
```

+++ {"id": "_tIN7ApEgyuU"}

Finally,  we need an initial guess to start the minimization. As with all minimization problems, this can be the most difficult step. It is always a good idea to use properties of the model and data where possible to make these guesses. We have no way to plot anything in four dimensions, so we use analysis instead.

We can derive some of these from the data we have. First, we can get the minimum in energy and the corresponding volume that we know from the data. These are not the final answer, but they are a good guess for it.

The B<sub>0</sub> parameter is related to the curvature at the minimum, which is the second derivative. We get that from repeated calls to `numpy.gradient`. Finally, $B_0'$ is related to the derivative of $B$ at the minimum, so we estimate that too.

```{code-cell} ipython3
:id: 23PtF7U-gyuU
:outputId: 5cdd057c-a093-4fc7-fd87-2250853e6a60

imin = np.argmin(energies)
dedv = np.gradient(energies, volumes)
B = np.gradient(dedv, volumes)
Bp = np.gradient(B, volumes)


x0 = [energies[imin], B[imin], Bp[imin], volumes[imin]]

x0
```

+++ {"id": "897-ybQ4gyuU"}

Finally, we are ready to fit our function. As usual, we also plot the data and the fit for visual inspection.

```{code-cell} ipython3
:id: 5ycsx3gQgyuV
:outputId: 15670d1b-a807-4944-ba64-ffb2f388369c

from scipy.optimize import minimize

sol = minimize(objective, x0)
print(sol)

plt.plot(volumes, energies, "bo", label="Data")
vfit = np.linspace(min(volumes), max(volumes))
plt.plot(vfit, Murnaghan(sol.x, vfit), label="fit")
plt.legend()
plt.xlabel("V")
plt.ylabel("E")
```

+++ {"id": "-NmPcWs8gyuV"}

That looks pretty good. We should ask ourselves, how do we know we got a minimum? We should see that the objective function is really at a minimum *for each of the parameters*. Here, we show that it is a minimum for the first parameter.

```{code-cell} ipython3
:id: wMJGnh8agyuV
:outputId: ed93e3ec-b0d5-42fd-88e8-6a36398acb25

E0_range = np.linspace(0.9 * sol.x[0], 1.1 * sol.x[0])

errs = [objective([e0, *sol.x[1:]]) for e0 in E0_range]

plt.plot(E0_range, errs)
plt.axvline(sol.x[0], c="k", ls="--")
plt.xlabel("E0")
plt.ylabel("summed squared error")
```

+++ {"id": "xu4s-POdgyuV"}

You can see visually that the error goes up on each side of the parameter estimate.

**exercise** Repeat this analysis for the other three parameters.

Later when we learn about linear algebra, we will learn that if you can show the eigenvalues of the Hessian of the objective function is positive definite, that also means you are at a minimum. It means the error goes up in any direction away from the minimum.

Usually we do some regression to find one of these:

1.  Parameters for the model - because the parameters mean something
2.  Properties of the model - because the properties mean something

In this particular case, we can do both. Some of the parameters are directly meaningful, like the E0, and V0 are the energy at the minimum, and the corresponding volume. B0 is also meaningful, it is called the bulk modulus, and it is a material property.

Now that we have a model though we can also define properties of it, e.g. *in this case* we have from thermodynamics that $P = -dE/dV$. We can use our model to define this derivative. I use `scipy.misc.derivative` for this for convenience. The only issue with it is the energy function has arguments that are not in the right order for the derivative, so I make a proxy function here that just reverses the order of the arguments.

```{code-cell} ipython3
:id: H5HIeXMRgyuV
:outputId: aab52dfb-847c-4a93-a5d3-aca02be473b7

from scipy.misc import derivative

pars = sol.x


def P(V):
    def proxy(V, pars):
        return Murnaghan(pars, V)

    dEdV = derivative(proxy, V, args=(pars,), dx=1e-6)
    return -dEdV


# Some examples
P(16), P(pars[-1]), P(18)
```

+++ {"id": "S_n7xWwbgyuW"}

The result above shows that it takes positive pressure to compress the material, the pressure is zero at the minimum, and it takes negative pressure to cause it to expand.

This example is just meant to illustrate what one can do with a model once you have it.

+++ {"id": "v8e-_RPlgyuW"}

## Parameter confidence intervals and curve fitting with `lmfit`

+++ {"id": "AJaB3pDIgyuW"}

We have left out an important topic in the discussion above: How certain are we of the parameters we estimated? This is a complicated question that requires moderately sophisticated statistics to answer. We discussed how to do this at the end of 06-262, and we used a special python package `lmfit` to do so!

`lmfit` has some nice properties:
* It's easy to use
* It's general for non-linear curve fitting problems
* It handled various loss functions
* It handles the basic statistics of linear uncertainty analysis for you

`````{seealso}
https://lmfit.github.io/lmfit-py/
`````

```{code-cell} ipython3
from lmfit import Model


def MurnaghanLmfit(vol, E0, B0, BP, V0):
    "From PRB 28,5480 (1983)"
    E = E0 + B0 * vol / BP * (((V0 / vol) ** BP) / (BP - 1) + 1) - V0 * B0 / (BP - 1.0)

    return E


gmodel = Model(
    MurnaghanLmfit, independent_vars=["vol"], param_names=["E0", "B0", "BP", "V0"]
)
params = gmodel.make_params(
    E0=energies[imin], B0=B[imin], BP=Bp[imin], V0=volumes[imin]
)

result = gmodel.fit(energies, params, vol=volumes)

print(result.fit_report())
vfit = np.linspace(min(volumes), max(volumes))

fitted_energies = result.eval(vol=vfit)


plt.plot(volumes, energies, "bo", label="Data")
plt.plot(vfit, fitted_energies)

plt.legend()
plt.xlabel("V")
plt.ylabel("E")
```

+++ {"id": "6uCxqIpPgyuZ"}

## Summary

+++ {"id": "uIGxSuAtgyuZ"}

Today we introduced the concept of finding minima/maxima in functions. This is an iterative process, much like finding the roots of a nonlinear function. You can think of it as finding the zeros of the derivative of a nonlinear function! This method is the root of many important optimization problems including regression.

`scipy.optimize.minimize` is the preferred function for doing minimization. There are other more specific ones described at [https://docs.scipy.org/doc/scipy/reference/optimize.html](https://docs.scipy.org/doc/scipy/reference/optimize.html), but `minimize` has a more consistent interface and provides almost all the functionality of those other methods.

Next time, we will look at how to apply minimization to regression problems.

We covered a lot of ground today. The key points are:

1.  Regression is a minimization of an accumulated error function.
2.  If you need uncertainty on the parameters from a regression, use `pycse.nlinfit`.
3.  If you need uncertainty on model predictions, you can either simulate it, or derive it. We will learn more about deriving it later.
