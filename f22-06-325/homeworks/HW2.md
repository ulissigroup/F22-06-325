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

+++ {"id": "YQvxPDWQDahl"}

# HW2 (due 9/12) 

## Global CO2 concentration prediction and practice with scikit-learn [100pt]

In class on Wednesday we showed that we can build predictive models for the northern hemisphere monthly average temperature. We achieved a MAE of about 0.5 C.

You're going to help me:
* Fit and predict the CO2 emissions from the Mauna Loa observatory in Hawaii
* analyze the temperature anomaly (temperature variation compared to the monthly average) instead of the absolute temperature
* use time series splits to analyze your model
* plot the temperature and your best fit

+++ {"id": "scEB7e1nIGGN"}

## Data download

Go to the Mauna Loa data website and find the link to the "Mauna Loa CO2 weekly mean and historical comparisons" text or csv file. Download it with wget.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: qMZMxnH-IBe4
outputId: 6cd4d228-6ea4-429f-be0c-a68b5c441da7
---

```

+++ {"id": "M-bb_vFjImJF"}

## Load and visualize the data

First, load the data with pandas. You will probably have to change the column names and the number of rows that are skipped compared to the example from class. Depending on whether you download the csv or the txt file you may also have to change delim_whitespace.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 488
id: vqu_0oM5JHcU
outputId: 623b5de5-d8e8-4632-d00e-7e7ee0995637
---

```

Now, plot the CO2 concentration in ppm vs the time (the decimal column in the sheet).

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 542
id: QA-_Lt8GJdGB
outputId: b7ed72ec-73ba-466a-a86b-055a65515423
---

```

## Filter the data

Note that some of the data is reported as -999.99. Those are days when there was no data at Mauna Loa for various reasons. Filter the dataframe to only include rows with positive ppm measurements. Repeat the plot from above to confirm the data looks good now!

`````{seealso}
https://www.google.com/search?q=pandas+filter+values+greater+than
`````

```{code-cell} ipython3

```

## Train/val-test split

To start, split your data into train/val (90%) and test (10%). Use `sklearn.model_selection.train_test_split` like we did in class. Make sure you don't shuffle when you do it, so that the test data is the most recent 10% of the data!

`````{tip}
`sklearn.model_selection.train_test_split` can split pandas dataframes directly!
`````

```{code-cell} ipython3

```

+++ {"id": "YLZHcggdKwBP", "tags": []}

# Your first scikit-learn model

Scikit-learn can handle all of the things we talked about in class! 
* Take data and featurizing it
* Implement and fit a machine learning model
* Create various train/test splits
* Calculate the average MAE/RMSE of the splits

To implement this, let's use a linear model with polynomial features, like we had in class!

`````{tip}
Helpful resources!
* https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
* https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline
* https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit
* https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py
`````

```{code-cell} ipython3
# This is the scikit-learn equivalent way of doing what we did in class. 
# I've include the code here to get you started. Look through the links above 
# for documentation on each of the pieces!

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# We expect a dataframe df_trainval generated above that contains the train and validation data
df_train, df_val = train_test_split(df_trainval, test_size=0.1, shuffle=False)

# Make a pipeline where we first generate quadratic polynomial features from the time data
# then fit using linear regression
model = make_pipeline(PolynomialFeatures(2), LinearRegression())

# Evaluate the model by generate 5 different train/val splits using TimseSeriesSplit,
# fitting the model from above on each train, and evaluating the MAE for the validation in each
cross_validate(
    model,
    df_train["decimal"].values.reshape(-1, 1), # The date in decimal format, X
    df_train["ppm"].values, # the CO2 concentration in ppm, y
    cv=TimeSeriesSplit(),
    scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
)
```

## Underfitting/overfitting

Vary the degree of the polynomial features above, and find the degree that minimized the mean absolute error on the final train/val split. Plot the MAE as a function of the polynomial degree.

```{code-cell} ipython3

```

```{code-cell} ipython3

```

## Visualize your best model

Now that you've optimized the degree of the polynomial to be most predictive for the train/val splits you identified, let's see how it does on the test data you set aside earlier!

Make a plot with plotly that shows:
* The train/val data and test concentration data
* The model you identified above, fitted on all of the train/val data and predicting the test date.
Include predictions for the next 5 years. Do these seem reasonable to you?

```{code-cell} ipython3

```

```{code-cell} ipython3

```

## Sources of bias and limitations

Discuss potential sources of bias or difficulties with this data? Is it possible to predict CO2 concentrations in the future without knowing if the world will take drastic action on CO2 emissions? Why or why not?

+++

## Bonus [10 pt]

We used quite simple features, and most likely your best fit has none of the annual cyclic variation present in the original dataset. 

Try incorporating strategies from one of these:
* Periodic spline features (https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#periodic-splines)
* https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#sphx-glr-auto-examples-applications-plot-cyclical-feature-engineering-py

To see if you can improve on the fit above. What's the lowest MAE you can obtain using the time series splits?

```{code-cell} ipython3

```
