---
jupytext:
  formats: ipynb,md:myst
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
This work was adapted from lecture notes in John Kitchin's excellent 06-623 course! His lecture notes are included in the helpful resources link if you want to know more details about how numerical methods work.
```

+++

# ODE Integration (with Events!)


`````{admonition} Lecture summary
:class: note
This lecture is going to:
* Quickly review how scipy.integrate.solve_ivp works
* Practice coding up a system of differential equations we covered extensively in 06-262
* Introduce how events works during ODE integrations (useful for future courses!)
* Demonstrate events for a more complicated non-linear higher order differential equation

Along the way, we will:
* Get used to the JupyterLab environment and the structure of the course notes
* Practice plotting in matplotlib
`````

+++ {"id": "sYhjkqf6WKfe", "tags": []}

## Review of `scipy.integrate.solve_ivp`

+++

As an example, consider the first order ODE:
\begin{align*}
y' = y + 2x - x^2; y(0) = 1
\end{align*}

This ODE has a known analytical solution: $y(x) = x^2 + e^x$. We will use this for comparison.

+++ {"id": "2kQMFOMGWKfe"}

The `scipy.integrate` library provides `solve_ivp` to solve first order differential equations. It is not the only one available, but this function is recommended. You import the function like this:

```{code-cell} ipython3
:id: Xz9BVvtwWKfe
:outputId: 520d4171-f524-402b-e09e-1e3d9d67cff2

import numpy as np
from scipy.integrate import solve_ivp
```

+++ {"id": "nx24SM4zWKfe"}

Here is a minimal use of the function, with keyword arguments.

`y0` is an array containing the initial values.  `fun` is a function with a signature of f(t, y). Here, $t$ is considered the independent variable. You can call it whatever you want, so f(x, y) is also fine. Since `solve_ivp` had $t$ in mind, the second argument is the `t_span`, which is a tuple of two numbers for where the integration starts (t0, or x0) and where it ends.  `solve_ivp` returns an object.

```{code-cell} ipython3
:id: XaEuU1aKWKfe
:outputId: a4213a09-30c6-4de3-a694-1c5c2b0d7541

def f(x, y):
    return y + 2 * x - x**2


x0 = 0
y0 = np.array([1])  # It is a good idea to make y0 an array. It will be important later.

sol = solve_ivp(fun=f, t_span=(x0, 1.5), y0=y0)
```

+++ {"id": "QctFd6OnWKfe"}

The output of `solve_ip` is an object containing results in attributes on the object.

```{code-cell} ipython3
:id: _OTEtvMPWKfe
:outputId: c4843f44-dac4-4f8a-813d-ee14a7511bec

sol
```

+++ {"id": "KvCfBovWWKfe"}

You should look for a few things here. One is that the message indicates success. Second, we access the solution using dot notation. Here are the independent variable values the solution was evaluated at.

```{code-cell} ipython3
:id: T1_CmSeDWKfe
:outputId: 59707552-1299-44d4-ec0e-e62498357449

sol.t
```

+++ {"id": "OEHyy77MWKff"}

Third, the solution is in a 2D array. We only have one equation here, so we use indexing to get the first row as an array.

```{code-cell} ipython3
:id: e3h4ETeYWKff
:outputId: 3f8f4c57-6fd9-4f03-af0a-b249e8c0423c

sol.y[0]
```

+++ {"id": "3_yOmHaCWKff"}

Now, we can plot the solution.

```{code-cell} ipython3
:id: eyYJt5_3WKff
:outputId: 366b0dad-8ac6-4da2-bfb5-0d2b5760cee1

import matplotlib.pyplot as plt

plt.plot(sol.t, sol.y[0], label="solve_ivp")
plt.plot(sol.t, sol.t**2 + np.exp(sol.t), "r--", label="Analytical")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
```

+++ {"id": "ITfM0WarWKff"}

That doesn't looks so great since there are only four data points. By default, the algorithm only uses as many points as it needs to achieve a specified tolerance. We can specify that we want the solution evaluated at other points using the optional `t_eval` keyword arg.

```{code-cell} ipython3
:id: 2gwGvLyZWKff
:outputId: 15c72153-762f-4480-cb58-ceae772c22e3

X = np.linspace(x0, 1.5)
sol = solve_ivp(fun=f, t_span=(x0, 1.5), y0=y0, t_eval=X)
print(sol)

plt.plot(sol.t, sol.y[0], label="solve_ivp")
plt.plot(X, X**2 + np.exp(X), "r--", label="Analytical")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
```

`````{admonition} **Tolerances** 
:class: note

solve_ivp is trying to estimate and control the error for you! 
* rtol is the relative tolerance in the function (eg % error in the numbers coming out). 
* atol is the absolute tolerance (I want the concentration +/- 0.00001). 
* rtol is $10^{-3}$ and atols is $10^{-6}$. 
* If your concentration is on the scale of 0.000000001 M, you will have a problem! 
  * Best solution - change units or rescale your problem so your variables are close to 1
  * Possible solution - make you atol really small and hope it solves things
* If decreasing rtol/atol changes your solution, they're not set tightly enough or you have other problems! 
`````

`````{admonition} **Integration failures**  
:class: note

The solve_ivp documentation has some nice comments for what to do it things go wrong with the default RK45 algorithm:

> If not sure, first try to run ‘RK45’. If it makes unusually many iterations, diverges, or fails, your problem is likely to be stiff and you should use ‘Radau’ or ‘BDF’. ‘LSODA’ can also be a good universal choice, but it might be somewhat less convenient to work with as it wraps old Fortran code
`````

+++

## ODE solving strategy review

In 06-262 we spent a lecture talking about how ODE integrators worked. We started with simpler Euler integration and worked our way up through Runge-Kutta, which you coded in one of the homework assignments. As a reminder, the Runge-Kutta update rules were a sequence of calculations that estimated the function value:
\begin{align}
k_1 &= \Delta t \cdot f(t^N, y^N)\\
k_2 &= \Delta t \cdot f\left(t^N+\frac{\Delta t}{2}, y^N + \frac{k_1}{2}\right)\\
k_3 &= \Delta t \cdot f\left(t^N+\frac{\Delta t}{2}, y^N + \frac{k_2}{2}\right)\\
k_4 &= \Delta t \cdot f(t^N+\Delta t, y^N + k_3)\\
y^{N+1} &= y^N + \frac{k_1}{6} + \frac{k_2}{3} + \frac{k_3}{3} +  \frac{k_4}{6}
\end{align}

We can watch how solve_ivp works by using function animations

```{code-cell} ipython3
def wrap_function_save_eval(t, y, function, function_eval_save):
    yp = function(t, y)
    function_eval_save.append((t, y, yp))
    return yp


x0 = 0
y0 = np.array([1])  # It is a good idea to make y0 an array. It will be important later.

# List to contain all of the function evaluations
function_eval_save = []

sol = solve_ivp(
    fun=wrap_function_save_eval, t_span=(x0, 1.5), y0=y0, args=(f, function_eval_save)
)
```

Now, we'll plot the final solution, as well as the intermediate evaluations as an animation

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation, rc

# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()
plt.close()

# Add the solution for solve_ivp
ax.plot(sol.t, sol.y.T, "ok--", label="solve_ivp solution")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim([0, 2])
ax.set_ylim([0, 10])

# Make a blank line and quiver to hold the data points as they get evaluated
(line,) = ax.plot([], [], "or", label="Function evaluations")
quiver = ax.quiver(x0, y0[0], 1, f(x0, y0)[0], color="r")
ax.legend()

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return (line,)

# animation function. This is called sequentially
def animate(i):
    
    # unzip the t, y, and yp vectors as separate vectors
    t, y, yp = zip(*function_eval_save)
    
    # Set the data for the line
    line.set_data(t[:i], y[:i])
    
    # Remove the old quiver and make a new one
    global quiver
    quiver.remove()
    quiver = ax.quiver(
        t[:i],
        y[:i],
        np.array(i * [0]) + 1,
        yp[:i],
        color="r",
        angles="xy",
    )
    return (line,)

# Make the animation!
anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=len(function_eval_save) + 1,
    interval=1000,
    repeat_delay=5000,
    blit=True,
)

# Note: below is the part which helps it work on jupyterbook
rc("animation", html="jshtml")
anim
```

`````{admonition} **Ask yourself these questions when solving ODE's!**
:class: tip

* Is my problem coupled or not? Higher order or not?
* Is my problem stiff? Should I use or do I need a special solver?
* Is there anything I can infer or guess about the solution from the differential  euqations that I can use to check my answers?
* How would I know if I made a mistake?
* Should there be a steady state? If so, how many steady states do I expect? 
* Is the answer reasonable? Do the units make sense?
* If you solve a problem with some tolerance or setting like $\Delta t = 0.1$ (or some other number like atol), always check that your answer does not change with $\Delta t = \frac{0.1}{2}$
* Before solving a problem with numerical methods, make sure you can correctly code the RHS of the equation.
`````

+++ {"tags": []}

## Using events during ODE integration

+++ {"id": "tIOg0zliWKff", "jp-MarkdownHeadingCollapsed": true, "tags": []}

So far, `solve_ivp` solves the issues with item 1 (we did not have to code the algorithm), and items 2 and 3 (it uses an adaptive step and converges to a tolerance for us). It will also help us solve for the inverse problem, i.e. for what value of $x$ is $y=4$?

To do this, we need a new concept of an "event function". During each step of the integration, you can run a function that can detect an "event". When an event is detected, the location of the event is stored, and if desired integration can be terminated. `solve_ivp` can take a list of event functions. We consider only one for now.

An event occurs when an event function is equal to zero. During integration, if the event function changes sign, then it is clear an event has occurred, and the algorithm determines where it occurred. Since we want to know when $y=4$, we will define a function that returns $y - 4$, because that will equal zero at that condition. We want the integration to terminate when that happens, so we set the "terminal" attribute on our function to True.

An event function has a signature of f(x, y). Remember that $y$ is going to be an array,

```{code-cell} ipython3
:id: a73bQ1u-WKff
:outputId: 5c4d2b64-1e99-4150-c0ba-cb8aace0e930

def event1(x, y):
    return y[0] - 4


event1.terminal = True

sol = solve_ivp(fun=f, t_span=(x0, 1.5), y0=y0, events=event1)
sol
```

+++ {"id": "QDJKunnLWKff"}

Now, there are a couple of new things to note. First, we got a message that a termination event occurred. Second, the sol.y array ends at 4.0, because we made the event function *terminal*. Next, sol.t\_events is not empty, because an event occurred. It now contains the value where the event occurred, which is where $y=4$!

```{code-cell} ipython3
:id: vLo7rh2xWKff
:outputId: 4d724dd4-36cd-44ab-abf6-61e5a1875d5a

sol.t_events[0]
```

```{code-cell} ipython3
:id: lbCHFLZ5WKfg
:outputId: cf6c60bd-cc85-4c9e-d50c-08662ff9abd3

sol.t
```

```{code-cell} ipython3
:id: 86jVTFLhWKfg
:outputId: 73f7e7d5-a93e-4ac7-bee8-b7ccc07f733b

print(f"y=4 at x={sol.t[-1]}. Confirming: y = {sol.t[-1]**2 + np.exp(sol.t[-1])}")
```

+++ {"id": "kr30mSOyWKfg"}

That is pretty close. You have to decide if it is close enough for the purpose you want. You can control the tolerance with optional `atol` and `rtol` keywords. You should read the documentation before changing this.

```{code-cell} ipython3
:id: ff4DU35uWKfg
:outputId: 12a96530-72f8-42f2-d796-bc0144495d24

def event1(x, y):
    return y[0] - 4


event1.terminal = True

sol = solve_ivp(fun=f, t_span=(x0, 1.5), y0=y0, events=event1, rtol=1e-9)
sol
sol.t[-1] ** 2 + np.exp(sol.t[-1])
```

We can also control the type of events that are considered by specifying whether the event is triggered in a specific direction. For example, we can find the first occurrence of $y=4$ coming from below

```{code-cell} ipython3
def event1(x, y):
    return y[0] - 4


event1.terminal = True
event1.direction = 1

sol = solve_ivp(fun=f, t_span=(x0, 1.5), y0=y0, events=event1, rtol=1e-9)
sol
sol.t[-1] ** 2 + np.exp(sol.t[-1])
```

## Summary

In addition to reviewing what we knew from 06-262 on `solve_ivp`, we also talked about how to use **integration events**. These allow us to stop integration at special points. It is very helpful for engineering/design problems.

`````{seealso}
More examples with ODE events are available at {doc}`./ode_events_extra_example`
`````

+++ {"tags": []}

## Practice: Lotka-Volterra  Review of Solving ODEs with Scipy!

+++

As a quick recap of where we left off in 06-262, let's start with an example we spent a lot of time covering, Lotka Volterra (rabbit/wolf) example.

We are interested in how the two populations of species (rabbits and wolves) might change over time. 

* Rabbits are $x(t)$, wolves are $y(t)$
* The rate of rabbits being eaten by wolves is proportional to both ($\beta xy$)
* Rabbits reproduce on their own at a rate proportional to the number of rabbits, and rabbits are eaten by wolves at a rate proportional to the number of rabbits and wolves 
\begin{align*}
\frac{dx}{dt}=\alpha x-\beta xy
\end{align*}
where $\alpha$ and $\beta$ are constants.
* Wolves are able to reproduce at a rate proportional to the number of wolves and rabbits (how quickly rabbits are being eaten) and die at a rate proportional to the number of wolves (sickness/injury/etc) 
\begin{align*}
\frac{dy}{dt}=\delta xy-\gamma y
\end{align*}

Let's say we start with 1 rabbit and 5 wolves, and the constants are 
* $\alpha=1 $[1/day]
* $\beta=0.2$ [1/wolves/day]
* $\delta=0.5$ [1/rabbits/day]
* $\gamma=0.2$ [1/day]

Let's take 10 minutes to warm up and try coding this up in python using (scipy)

+++

### Solve for and plot the population of rabbits and wolves over the first 20 days

```{code-cell} ipython3

```

### Find the the first occurrence that the rabbit population hits 3

```{code-cell} ipython3

```

### Find the the first occurrence that the wolf population hits 3 while the wolf population is growing

```{code-cell} ipython3

```
