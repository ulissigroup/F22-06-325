---
jupytext:
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

```{sidebar} Adaptation!
This work was copied from John Kitchin's 06-623 course! It is used as a test case.
```

+++ {"id": "mJnRxC7iI5UF"}

# Higher order differential equations

+++ {"id": "9qgQVg-OI5UH"}

So far we have focused on computational solutions to first order differential equations, including systems of first order differential equations. The reason for that is simply that all numerical integration strategies only work with the first derivative.

Many differential equations involve higher order derivatives though. We can solve these by converting them to systems of first-order differential equations through a series of variable changes.

Let's consider the [Van der Pol oscillator](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator).

$\frac{d^2x}{dt^2} - \mu(1-x^2)\frac{dx}{dt} + x = 0$

We define a new variable: $v = x'$, and then have $v' = x''$.

That leads to a set of equivalent first-order differential equations:

$x' = v$

$v' - \mu (1-x^2)v + x = 0$

You can still think of $x$ as the position of the oscillator, and $y$ as the velocity of the oscillator. Now, we can integrate these equations from some initial condition.

Let's do this and plot the position and velocity of the oscillator. Rather than use `t_eval`, we will instead set the optional argument `max_step` to tell the solver how often it should make a step.

This is different than using `t_eval`, which uses interpolation *after* the solution has been found to evaluate the solution. This will become important later when we use events, which are only evaluated at the *solver* points.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
executionInfo:
  elapsed: 194
  status: ok
  timestamp: 1631795105482
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: yyrECic4I5UI
outputId: b30eb639-b85d-475a-e847-12fc39851bb3
---
import numpy as np
from scipy.integrate import solve_ivp

mu = 0.2


def dXdt(t, X):
    x, v = X
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return np.array([dxdt, dvdt])


X0 = np.array((1, 2))  # you can pick any x0, and v0 you want.
tspan = np.array((0, 40))

dXdt(0, X0)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
executionInfo:
  elapsed: 137
  status: ok
  timestamp: 1631795106455
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: TXZoCS_z9qAS
outputId: f5a93ece-08b3-4073-db06-6423d88def02
---
teval, h = np.linspace(*tspan, 500, retstep=True)

sol = solve_ivp(dXdt, tspan, X0, max_step=h)
sol.message, sol.success, sol.y.T.shape
```

+++ {"id": "oR6BcTJ8I5US"}

Now, we can plot the solutions.


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 279
executionInfo:
  elapsed: 419
  status: ok
  timestamp: 1631795107931
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: p3elVIq_I5UT
outputId: ea6feb21-3b60-4c42-fde0-f15fa9d0430f
---
import matplotlib.pyplot as plt

plt.plot(sol.t, sol.y.T)
plt.xlabel("t")
plt.ylabel("x,v")
plt.legend(["x", "v"]);
```

+++ {"id": "7aSzbBrkI5Ua"}

You can see that the solution appears oscillatory. Let's be more quantitative than what it *looks* like. An alternative way to visualize this solution is called the phase portrait where we plot the two state variables (x, v) against each other. We include the starting point for visualization.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 279
executionInfo:
  elapsed: 324
  status: ok
  timestamp: 1631795185365
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: P49UZdKjI5Ub
outputId: 5ccdc373-b74f-480a-add0-8ac16e29fba3
---
plt.plot(*sol.y)  # unpack the first row into x and second row into the y
# equivalent to plt.plot(sol.y[0], sol.y[1])
plt.plot(*sol.y[:, 0], "go")  # starting point
plt.xlabel("x")
plt.ylabel("v")
plt.axis("equal");
```

+++ {"id": "xacH7-h9I5Uh"}

So, evidently it is not exactly periodic in the beginning, but seems to take some time to settle down into a periodic rhythm. That seems to be the case, because if it didn't we would expect to see a continued spiral in or out of this limit cycle. Another way we can assess this quantitatively is to look at the peak positions in our solution. We return to an event type of solution. We seek an event where the derivative $dx/dt=0$, and it is a maximum, which means $x'$ starts positive, becomes zero, and then is negative. Note this is appropriate for this problem, where there is only one, periodic maximum. For other problems, you might need a different approach.

Now, it is important to remember that the event function is only evaluated after a solver point, so we need to make sure the solver points bracket where events occur. This is accomplished by making sure that when we graph the solution from the solver (not from t_eval), that we can visually see where the events will occur.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
executionInfo:
  elapsed: 140
  status: ok
  timestamp: 1631795784541
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: zR0IFJnlI5Ui
outputId: bc4f88b0-3980-49b7-f2a9-0024643e2747
---
def max_x_event(t, X):
    Xprime = dXdt(t, X)
    return Xprime[0]  # first derivative, dx/dt = 0


max_x_event.direction = -1  # event must go from positive to negative, i.e. a max

sol = solve_ivp(dXdt, tspan, X0, max_step=h, events=max_x_event)
print(sol.message)
sol.t_events[0]
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
executionInfo:
  elapsed: 149
  status: ok
  timestamp: 1631795786589
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: OIVR3oJRc7xl
outputId: 769ed4eb-2b76-49ad-bc7a-eb7bb4f088ec
---
print(sol.y_events[0])
```

+++ {"id": "j5WO37hwI5Uo"}

You can see we found seven events. We should plot them on the solution to see that they are in fact maxima. (what could possibly go wrong? if you get the wrong direction, then you will either see minima, or minima and maxima! If your event function is wrong, then it will just be wrong.)  Note we get two rows in our solution, one for x and one for v. From the numbers here, you can see that the x_max values seem to be settling down to about 2.0.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 279
executionInfo:
  elapsed: 433
  status: ok
  timestamp: 1631795793227
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: RhFB_mD7I5Uv
outputId: b14775b4-94a5-45af-d3b3-71751b557f53
---
plt.plot(sol.t, sol.y.T)  # full solutions

# break up this calculation for ease of reading
te = sol.t_events[0]
xmax, v_at_xmax = sol.y_events[0].T
plt.plot(te, xmax, "ro")
plt.plot(te, v_at_xmax, "bo")

# compare to. Don't do this, it is confusing and hard to figure out.
# plt.plot(sol.t_events[0], sol.y_events[0][:, 0], 'k*')

plt.xlabel("t")
plt.ylabel("x,v")
plt.legend(["x", "v"]);
```

+++ {"id": "fsMXDtD-I5U1"}

That looks good, the red dots appear at the maxima, and they are periodic, so now we can see how x<sub>max</sub> varies with time.


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 296
executionInfo:
  elapsed: 615
  status: ok
  timestamp: 1631795816341
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: tdaHCQgSI5U3
outputId: 34624dee-8963-4dc5-a4c1-9b4ce54affe9
---
plt.plot(te, xmax, "ro")
plt.xlabel("t")
plt.ylabel("$x_{max}$");
```

+++ {"id": "AbjhK69cI5U8"}

You can see that after about 5 cycles, xmax is practically constant. We can also see that the period (the time between maxima) is converging to a constant. We cannot say much about what happens at longer times. You could integrate longer if it is important to know that. This is a limitation of numerical methods though. To *prove* that it will be constant, you need to do some analytical math that would show the period and x<sub>max</sub> go to a constant.


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 279
executionInfo:
  elapsed: 444
  status: ok
  timestamp: 1631795945118
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: H9BjhQYBI5U9
outputId: f02a0290-cdbb-43bb-98f9-27f50bf21049
---
plt.plot(np.diff(te), "bo")
plt.xlabel("cycle")
plt.ylabel("period");
```

+++ {"id": "kFX7lct0I5VD"}

If we seek the steady state, oscillatory behavior of this system, we should discard the solutions in at least the first 4 cycles, since the maxima and periods are still changing.


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
executionInfo:
  elapsed: 129
  status: ok
  timestamp: 1631796032024
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: 6oOUIckyI5VE
outputId: 75de1078-52e8-4b1a-a16f-b08a3475c764
---
te[-1], sol.y_events[0][-1]
```

+++ {"id": "5DRqVFEAI5VJ"}

Alternatively, we can use the last point as an initial value for a new integration that should be close to steady state oscillations.


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 279
executionInfo:
  elapsed: 538
  status: ok
  timestamp: 1631796052397
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: tGmhmmk6I5VK
outputId: 16c9ef84-374e-4b48-9e02-306f7cda58e1
---
tspan = (0, 40)
X0 = sol.y_events[0][-1]

sol2 = solve_ivp(dXdt, tspan, X0, max_step=h, events=max_x_event)
plt.plot(sol2.t, sol2.y.T)
plt.xlabel("t")
plt.ylabel("x,v");
```

+++ {"id": "mzhcuMjlI5VQ"}

Here you see about 6 more cycles. The period of these events is practically constant.


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
executionInfo:
  elapsed: 119
  status: ok
  timestamp: 1631796062174
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: cHJlzpD5I5VR
outputId: 769e20be-de06-4baa-f078-3723073f1a31
---
sol2.t_events, np.diff(sol2.t_events[0])
```

+++ {"id": "OWLApvT8I5VX"}

And the limit cycle shows practically a single curve.


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 279
executionInfo:
  elapsed: 458
  status: ok
  timestamp: 1631796065171
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: dNXqXui-I5VY
outputId: 6dd8e1fe-8077-4ecd-e903-0fdc48a0433e
---
plt.plot(*sol2.y)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
# makes x-ticks have the same dimension as y-ticks
```

+++ {"id": "bbaZa1aSI5Vd"}

This limit cycle shows the oscillatory behavior. You can see here that each cycle repeats on top of itself.

**Review** We have been working on finding a steady state oscillatory solution to $\frac{d^2x}{dt^2} - \mu(1-x^2)\frac{dx}{dt} + x = 0$, which describes an oscillating system. We examined some ways to tell if a system is oscillating, and to estimate the period of the oscillation.


```{code-cell} ipython3
---
executionInfo:
  elapsed: 119
  status: ok
  timestamp: 1631796236856
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: Z3vm_Tx1F3OZ
---
?solve_ivp
```

+++ {"id": "51_iHp7bI5Ve"}

### Solving a parameterized ODE many times


+++ {"id": "2Q3xwyW6I5Vf"}

$\mu$ in the Van der Pol system is called a parameter. It is common to study the solution of this system as a function of &mu;. For [example](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator#/media/File:VanderPol-lc.svg), the oscillatory behavior changes a lot as &mu; changes. Our aim here is to recreate the figure in that example, showing the steady state limit cycles as a function of &mu;.

The example we want to create has limit cycles for 10 different values of &mu;. *We do not want to copy and paste code 10 times*. Instead, we should have some code we can *reuse 10 times*.

Let's break this task down. For a given &mu;, we should find a solution to the ODEs that shows constant periods. That means we should integrate over a time span, check the periods, and if they are not constant, integrate from the last point over the time span again. If they are consistent, then we can just plot the solution.

How can we check the periods are constant? One way is to see if the first and last are the same within some tolerance, say 1e-3.

Ideally, we would have a function that takes one argument, &mu;, and returns the steady state oscillatory solution.

```{code-cell} ipython3
---
executionInfo:
  elapsed: 118
  status: ok
  timestamp: 1631797272235
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: UjpIg5k1I5Vh
---
# We do not have to define this here, I just repeat it so you can see it again.
def max_x_event(t, X):
    x, v = X
    Xprime = dXdt(t, X)
    return Xprime[0]  # first derivative = 0


max_x_event.direction = -1  # event must go from positive to negative, i.e. a max


def get_steady_state(mu):
    # define the sys odes for this mu. We define it inside the function so it
    # uses the mu passed in to get_steady_state.
    def dXdt(t, X):
        x, v = X
        dxdt = v
        dvdt = mu * (1 - x**2) * v - x
        return np.array([dxdt, dvdt])

    X0 = np.array([2, 0])  # start at x_max, velocity=0

    tspan = np.array([0, 40])  # we assume we will get 4-6 periods this way
    teval, h = np.linspace(*tspan, 1500, retstep=True)

    # initial solution
    sol = solve_ivp(dXdt, tspan, X0, max_step=h, events=max_x_event)
    periods = np.diff(sol.t_events[0])

    # Now iterate as long as the first and last periods differ by more than the
    # tolerance. It is usually a good idea to provide a way to break out in case
    # it never ends. Here we use a max iteration count.
    i = 0

    # This assumes there are at least 2 periods in the tspan.
    while np.abs(periods[0] - periods[-1]) > 1e-3:
        last_step = sol.y[:, -1]  # this is the new initial condition to continue from.
        sol = solve_ivp(dXdt, tspan, last_step, max_step=h, events=max_x_event)
        # now get new periods.
        periods = np.diff(sol.t_events[0])
        i += 1  # increase the counter
        if i > 5:  # if we exceed 5 iterations, something is probably wrong, so stop.
            dp = np.abs(periods[0] - periods[-1])
            print(dp, periods)
            print(f"Max iterations exceeded and no stability for mu={mu}")
            break
    print(f"For mu={mu}, steady period after {i} iterations")

    # Finally, return the last solution
    return sol
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 282
executionInfo:
  elapsed: 642
  status: ok
  timestamp: 1631797392787
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: Lc04jFwHIl0y
outputId: 12092c98-097f-4eee-8739-c466ed16612e
---
sol = get_steady_state(0.2)
plt.plot(*sol.y)
plt.axis("equal");
```

+++ {"id": "_L-4MzquI5Vm"}

Note: This takes about a second per iteration to run.


```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 544
executionInfo:
  elapsed: 4964
  status: ok
  timestamp: 1631797623885
  user:
    displayName: John Kitchin
    photoUrl: https://lh3.googleusercontent.com/a/default-user=s64
    userId: '14782011281593705406'
  user_tz: 240
id: mCoqfCFFI5Vn
outputId: ba8135c3-6813-4c74-daf6-2b0c35af0787
---
MU = [0.01, 0.1, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
MU.reverse()  # in place reversal

plt.figure(figsize=(3, 6))  # makes a figure that is ~3 inches wide and 6 inches high
for mu in MU:
    sol = get_steady_state(mu)
    plt.plot(
        *sol.y, lw=0.1, label=f"{mu:1.2f}"  # Use thinner linewidth, default=1
    )  # defines a label for the legend

plt.legend(
    title="$\mu$",
    loc="upper center",
    # this line says put the legend outside the box.
    # (0, 0) is the lower left, (1, 1) is the upper right
    bbox_to_anchor=(1.2, 1),
)

plt.axis("equal");
```

+++ {"id": "-nYCrqJzI5Vs"}

## Summary


+++ {"id": "8iGh-rFQI5Vu"}

Today we covered the conversion of an n<sup>th</sup> order differential equation into a system of first order differential equations.

We examined the use of the optional argument max\_step to fine tune the solution points returned by the solver.

This concludes our first section on ordinary differential equations.
