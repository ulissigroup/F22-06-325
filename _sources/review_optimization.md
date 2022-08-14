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
This lecture was also adapted from lecture notes in John Kitchin's excellent 06-623 course! His lecture notes are included in the helpful resources link if you want to know more details about how numerical methods work.
```

+++

# Review of Numerical Methods for Local Optimization


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

import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

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

What are the pros and cons of this method:

Pros:

1.  It is *easy*.
2.  You *see* the whole domain you are looking at, and it is easy to see how many extrema their are

Cons:

1.  *Lot's* of function evaluations. Imagine if it took a long time to compute each value.
2.  Somewhat tedious.
3.  Not so easy to reproduce
4.  Not scalable to large problems, your time to do this becomes a limiting factor.

+++ {"id": "Y-WUmY0Odsgi"}

### Find the derivative, and solve for where it is zero

+++ {"id": "0KYh-8qkdsgi"}

We can also derive the first derivative:

$y' = 2 * x + e^{-5 x^2} (-10 * x)$

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

+++ {"id": "bpiWX6jwdsgk"}

### Newton-Raphson method of minima finding

+++ {"id": "EltrbTpNdsgk"}

To use the Newton-Raphson method to get the minimum, we use an iterative approach with:

$x_{n+1} = x_n - \frac{y'(x_n)}{y''(x_n)}$.

We have to derive these formulas if you want to use analytical derivatives:

$y' = 2 * x + e^{-5 x^2} (-10 * x)$

$y'' = 2 + e^{-5 x^2} (-10 * x)^2 - 10 e^{-5 x^2}$

Alternatively, we can estimate the derivatives numerically using `scipy.misc.derivative`. This has the downside of numerical instability for dx that is too small, or low accuracy if it is too large, and the need to check if you made a good choice for it. On the plus side, it avoids making mistakes in the derivative derivation and implementation.

```{code-cell} ipython3
:id: j-_RUWG9dsgk
:outputId: 98abb7cb-4f29-4b10-e248-2fa0a8b31160

from scipy.misc import derivative

x0 = 0.2
f0 = f(x0)

for i in range(15):
    yp = derivative(f, x0, dx=1e-6, n=1)
    ypp = derivative(f, x0, dx=1e-6, n=2)
    xnew = x0 - yp / ypp
    fnew = f(xnew)

    if np.abs(yp) <= 1e-6:
        break
    x0 = xnew
    f0 = fnew

xnew, fnew, yp, i
```

+++ {"id": "FWnxfwdedsgl"}

This answer also agrees to at least 5 decimal places. This is the gist of what happens in fsolve.

As we have seen many times, finding minima is such a common task that there are dedicated functions available for doing it. One of the is `scipy.optimize.fmin`. This has a similar signature as `scipy.optimize.fsolve`, you give it a function and an initial guess, and it iteratively searches for a minimum.

+++ {"id": "tM3Sub3Cdsgl"}

## scipy.optimize.minimize

```{code-cell} ipython3
:id: UYKDxB5Wdsgl
:outputId: 0b1154d9-976e-493c-fe55-60644b006b2e

from scipy.optimize import minimize
minimize?
```

+++ {"id": "gVbU1KXjdsgl"}

Here is the basic use of fmin. As always, we should plot the answer where feasible to make sure it is the minimum we wanted.

```{code-cell} ipython3
:id: mJDxo14Adsgm
:outputId: a1121bee-cb03-41f3-e55c-dadb66c1549a

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

plt.plot(x, y, 'b-')
plt.plot(sol.x, f(sol.x), 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['f(x)', 'fmin'])
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
    return 2 + np.cos(x) + np.cos(2*x - 0.5) / 2

x = np.linspace(0, 2 * np.pi)

plt.plot(x, h(x))
plt.xlabel('x')
plt.ylabel('h(x)')
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

+++ {"id": "KDFDbY0Adsgn"}

### Finding maxima

+++ {"id": "fx2z_UFcdsgn"}

`fmin` is for finding *minima*. We can use it to find maxima though, but finding the *minima* of $-f(x)$. You can see here that when we plot $-h(x)$ the minima become maxima, and vice-versa. Now you can see there are two definite minima, one near zero, and one near 3.5, which correspond to the maxima of $h(x)$.

```{code-cell} ipython3
:id: 8dZFxsEidsgn
:outputId: 51d00a2a-7373-4f18-a8ef-1b875316cf83

plt.plot(x, -h(x))
plt.xlabel('x')
plt.ylabel('-h(x)')
```

+++ {"id": "1QXm-JUzdsgn"}

The standard way to use fmin is to define an optional argument for the sign that defaults to one. Then, when we call fmin, we will pass -1 as the sign to the function, so we find the minimum of -h(x). Then, we evaluate h(x) at that x-value to get the actual value of the maximum. It is not necessary do this, you can also manually pass around the sign and try to keep it straight.

Here is an example to find the maximum near 3.5.

```{code-cell} ipython3
:id: XlR0j1Drdsgn
:outputId: 3a96db79-cb3e-4d08-a198-8516940dbbea

def h(x, sign=1):
    return sign * (2 + np.cos(x) + np.cos(2*x - 0.5) / 2)

sol = minimize(h, 3.5, args=(-1,))  # set sign=-1 here to minimize -h(x)
print(h(sol.x))  # sign defaults to 1 here, so we get the maximum value

plt.plot(x, h(x))
plt.plot(sol.x, h(sol.x), 'ro')
plt.xlabel('x')
plt.ylabel('h(x)')
```

+++ {"id": "IHKDpXeedsgn"}

Once again, here you have to decide which maximum is relevant

+++ {"id": "gXyE36TCdsgn"}

### Application to maximizing profit in a PFR

+++ {"id": "JAYePfYNdsgo"}

Compound X with concentration of $C_{X0} = 2.5$ kmol / m<sup>3</sup> at a flow rate of 12 m<sup>3</sup>/min is converted to Y in a first order reaction with a rate constant of 30 1/min in a tubular reactor. The value of Y is $1.5/kmol. The cost of operation is $2.50 per minute per m<sup>3</sup>. Find the reactor length that maximizes the profit (profit is value of products minus operating costs).

First, consider why there is a maximum. At low volumes the operating cost is low, and the production of Y is low. At high volumes, you maximize production of Y, so you have the most value, but the operating costs go up (to infinity for complete conversion!). Somewhere in the middle is where a maximum is.

Here are some relevant constants.

```{code-cell} ipython3
:id: H88WBDpUdsgo
:outputId: a7fd9f66-aadf-4900-bc4f-75e4df39d7f6

cost = 2.5  # dollar/min/m**3
y_value  = 1.5 # dollar / mol

Cx0 = 2.5 # kmol / m**3
v0 = 12.0 # m**3 / min

k = 30.0 # 1/min
```

+++ {"id": "x_JosbU-dsgo"}

To compute the profit as a function of reactor volume, we need to compute how much Y is produced, then multiply that by the value of Y and subtract the operating cost. To compute how much Y is produced, we use a mole balance on X and Y, and integrate it to the volume to get the molar flows of X and Y. I like to write mole balances like this.

```{code-cell} ipython3
:id: 8NLNbeUqdsgo
:outputId: 70cc0c2a-8183-443e-cb93-74ef2b20208c

def dFdV(V, F):
    'PFR mole balances on X and Y.'
    Fx, Fy = F
    Cx = Fx / v0
    rx = -k * Cx
    ry = -rx

    dFdX = rx
    dFdY = ry
    return [dFdX, dFdY]

F0 = [Cx0 * v0,  # Fx0
      0.0]       # Fy0
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
plt.xlabel('V')
plt.ylabel('profit')
```

+++ {"id": "VRHIDvV9dsgo"}

You can see from this plot there is a maximum near V=1.5. We can use that as a guess for fmin.

```{code-cell} ipython3
:id: BbOkzYrhdsgo
:outputId: 501165f5-fb45-4464-8c9f-7dfbfd64d8a7

from scipy.optimize import fmin
sol = minimize(profit, 1.5, args=(-1,))

print(f'The optimal volume is {sol.x[0]:1.2f} m^3 with a profit of ${profit(sol.x[0]):1.2f}.')
```

+++ {"id": "a3QJdbUxdsgo"}

This problem highlights the opportunities we have to integrate many ideas together to solve complex problems. We have integration of an ODE, nonlinear algebra/minimization, with graphical estimates of the solution.

**Practice** Can you solve this with an event and solve\_ivp?

+++ {"id": "oDcqY0qadsgo"}

## Summary

+++ {"id": "NMnoULwydsgp"}

Today we introduced the concept of finding minima/maxima in functions. This is an iterative process, much like finding the roots of a nonlinear function. You can think of it as finding the zeros of the derivative of a nonlinear function! This method is the root of many important optimization problems including regression.

`scipy.optimize.minimize` is the preferred function for doing minimization. There are other more specific ones described at [https://docs.scipy.org/doc/scipy/reference/optimize.html](https://docs.scipy.org/doc/scipy/reference/optimize.html), but `minimize` has a more consistent interface and provides almost all the functionality of those other methods.

Next time, we will look at how to apply minimization to regression problems.

```{code-cell} ipython3

```
