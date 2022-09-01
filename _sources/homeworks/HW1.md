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

$$\require{mhchem}$$

+++

# HW1 (due 9/4)

+++ {"tags": []}

# Catalyst deactivation in a batch reactor [50 pts]

Consider the irreversible, liquid-phase isomerization reaction carried out in a solvent containing dissolved catalyst at 25 C in a batch reactor:
\begin{align*}
\ce{B ->[k_b] C}
\end{align*}
The apparent first-order reaction rate constant $k_b$ decreases with time because of catalyst deterioriation. A chemist friend of yours has studied the catalyst deactivation process and has proposed that it can be modeled with
\begin{align*}
k_b = \frac{k}{1+k_dt}
\end{align*}
in which $k$ is the fresh catalyst rate constant and $k_d$ is the deactivation rate constant.

We can derive the ODE to solve by starting with a mol balance on the entire system:
\begin{align*}
\frac{dN_b}{dt} &=N_b^0 - Vr \\
\frac{dC_b}{dt} &= -k_bC_b=-\frac{kC_b}{1+k_dt} 
\end{align*}
with the initial condition $C_b(t=0)=C_{b0}=5$ M.

+++

## Mol balance solve

Solve the mole balance for $C_B(t)$ assuming $k$=0.6/hr and $k_d$=2/hr for the first two hours. Plot the conversion % for your solution (defined as $1-C_B(t)/C_{B0}$).

```{code-cell} ipython3

```

## If it takes two hours to reach 50% conversion and the fresh catalyst has a rate constant of 0.6/hr what is the actual $k_d$?

```{code-cell} ipython3

```

## Using $k_d$ you found from the previous step, use `solve_ivp` events to determine how long it takes to reach 75% conversion in the reactor.

```{code-cell} ipython3

```

## Catalyst refresh
Say that we can stop the batch process after 2 hours, filter the spent catalyst, and replace with fresh catalyst. $C_B$ will start wherever the first reaction left off. Solve and plot for $C_B(t)$ over 5 hours, and include the plot with the catalyst refresh. Include a legend on the plot to label each solution

```{code-cell} ipython3

```

+++ {"tags": []}

# Curve fitting with two species and multiple data (50 pt)

Consider the irreversible reaction:
\begin{align*}
\ce{A + B -> Products}
\end{align*}
with $r=kC_A^nC_B^m$ taking place in an isothermal liquid-phase batch reactor. Measurements of $C_A$ vs $C_B$ are included in the attached file isothermal_liquidphase_batch_data.dat. We wish to determine from the data the rate constant and the order of the reaction with respect to A and B.  We have data from two experiments.

+++

## Load the data from the file into a numpy array and plot the concentration of each species

You can use either the csv library https://docs.python.org/3/library/csv.html or pandas https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html. 

The first column is time in minutes. The second and third column is C_A and C_B for the first experiment (in mol/L). The fourth and fifth column is C_A and C_B for the second experiment. Plot the data for $C_A$ and $C_B$ for each experiment (one experiment per figure).

```{code-cell} ipython3

```

+++ {"tags": []}

## Using lmfit, estimate rate parameters $k, n, m$ and initial concentration $C_{A0},C_{B0}$ from the data in the first experiment using a numerical solution for the concentrations of each species (standard mol balance + odeint).  Plot the final fit of the experiment along with the experimental data and calculate the uncertainty in each value

```{code-cell} ipython3

```

## Consider the second experiment, estimate the parameters using only this data. Comment on how the values and confidence intervals are different than the first case

```{code-cell} ipython3

```

## Estimate the parameters using both experiments simultaneously. Are the confidence intervals better? Comment on why or why not.

```{code-cell} ipython3

```

## Using the noise-free data in isothermal_liquidphase_batch_errorfree.dat, repeat the fit using all of the data. Do the results agree with your initial trials?

+++

## Based on these results, to determine the order of the rate expression with respect to two different species, how should you choose the initial conditions of the two species?

```{code-cell} ipython3

```
