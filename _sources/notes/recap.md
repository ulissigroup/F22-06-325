# Recap of Numerical Methods

We covered some basic numerical methods alongside the math in 06-262. 

In linear algebra, we covered:
* numpy and arrays
* matrix algebra (multiplication, inverses, etc)
* eigenvalues/eigenvectors

In differential equations we covered:
* Core ideas behind ODE integration schemes and things to watch out for (step size, stiff systems, etc)
* Solving first order and systems of ODEs using `scipy.integrate.solve_ivp`
* Solving higher order ODEs by turning them into ODEs
* Plotting solutions in 2d
* Calculating steady states of a system with scipy.optimize.fsolve
* Visualizing solutions to transient PDEs

Finally, we did some basic statistics and regression including:
* Linear regression (by hand and using `statsmodels`)
* Non-linear regression (using `lmfit`)
* Parameter estimation and confidence intervals

You demonstrated that you understood these ideas with your final project. Since these ideas are very important and you've had a summer break since then, we're going to spend the first week of class reviewing some of this material. The review of optimization and regression will be particularly important as we'll leverage many of the same ideas when we turn to linear regression!
