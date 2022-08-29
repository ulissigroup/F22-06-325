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

+++ {"id": "mJnRxC7iI5UF", "tags": []}

## More complicated example: Van der Pol oscillator

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
