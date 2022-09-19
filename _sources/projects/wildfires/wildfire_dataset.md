---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst
  main_language: python
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

![wildfire](./prescribed_burn_public_domain.jpg)

# Predicting Wildfire Smoke Composition

Prof. Jen is a world-expert at understanding the composition and wildfire smokes. In 2017, she was part of an experimental campaign to map the composition of smokes for controlled wildfire burns for several specific plots of forest at the Blodgett Forest Research Station in Georgetown, CA. 

Prof. Jen and her collaborators exposed filters to the burns at either ground level or at elevation using remote-controlled drones (each drone had three filters). They then took those filters and used a special analytic technique (GCxGC/MS, which you probably learned about in analytical chem) to identify unique spectral signatures of compounds present in the filters. In a few cases they know the compounds that generate specific signatures, but in many cases it's unclear exactly which compound led to a specific GCxGC/MS signature. 

`````{note}
**Wildfire dataset summary:**
* 3 different plots of land (with labels 60, 340, 400) were burned. One unburned plot was also included as a control (0).
* Each plot was sampled multiple times at varying times. 
    * Plots were sampled at the ground level in triplicate (3 filters)
    * Plots were sampled with drones at elevation in triplicate (3 filters)
* All filters were collected and analyzed with GCxGC/MS. The unique ID of blobs present and the associated concentration on the filter were recorded. 
* The prevalent plants and foliage present in each plot is also known based on a previous survey of the regions. 
`````

`````{seealso}
You can read more about how one of Prof. Jen's collaborators analyzed this data [here](https://www.climatechange.ai/papers/neurips2020/82). That same site includes both a paper and a short video by a collaborator on the specific analysis tried.
`````

## Suggested challenges

* Given a filter and a set of observed blobs, predict whether that filter was exposed at ground level or at elevation (with a drone)
* Given the filter of a filter at elevation (drone, easy to collect data), predict the blobs and their concentrations for the ground level measurements (harder to collect data)
* [much harder] Given the filter and a set of observed blobs, predict the types of plants present in the plot of land

````{note}
In each case, you should think critically about the question how you want to set up your train/validation/test splits to simulate the challenge. 
* What do you expect to be correlated?
* How would you know if your model is helpful?
* Is there any way to cheat at the task? 
````

## Dataset download/availability instructions
https://github.com/ulissigroup/F22-06-325/tree/main/f22-06-325/projects/wildfires/data

## Dataset/file format details
1. BlodgettCombinedBlobMass.csv is a spreadsheet that gives the electron ionization mass spectrum for each compound detected during the Blodgett field campaign. 
    * The mass spectrum (each element) is written as mass, signal; mass, signal; etc.
    * The row number corresponds to the compound of the same row number found in BlodgettCombinedBlobTable.csv
2. BlodgettCombinedBlobTable.cvs contains all compound 2 dimension gas chromatography data from all samples collected from Blodgett 2017. The column headings are:
    1. Unused tag
    2. BlobID_1
    3. Unused Tag
    4. 1D retention time (min)
    5. 2D retention time (sec)
    6. Peak height
    7. Peak volume
    8. Peak volume divided by nearest internal standard peak volume
    9. Calculated d-alkane retention index
    10. matched retention index (this number should be super close to the retention index in column 9)
    11. Unused tag
    12. Unused tag
    13. Unused tag
    14. BlobID_2
    15. Filter number. This is the filter number that can be linked to where and when the sample was collected
    16. Unused tag
    17. Mass concentration of this compound (ng/m3)
    * BlobID_1 and BlobID_2 (column 2 and 14) define the unique ID of a blob that can be tracked across the different burns. In other words, a compound (blob) with an ID of 1,176 is the same compound in filter 201 and filter 202. 
    * The d-alkane retention index (column 10) and 2nd dimension retention time (column 5) define the unique x,y position the compound sits in the chromatogram. No two compounds will have the same x,y coordinate. 
    * Mass concentration defines the amount of compound that exists in the smoke. 
3. Run Log.xlsx details where each filter was collected in Blodgett by GPS location and forest plot that burned. Tab “Flight Log” provides the details of filters collected from the drone. Tab “ground Station” provides details of filters collected at ground level.
4. All_Shrubcovony_01_16.xlsx displays the types of shrubs that grew at Blodgett. The sheet of interest is “16” which stands for 2016 when they conducted a plant inventory. Focus on Unit (1st column) 60, 340, and 400 which stand for the plots that we burned at Blodgett. Species column lists the shorthand for the shrub/grass that they observed growing in the plot. BFRSPlantCodes.xlsx translate the shorthand plant code to a real plant.
5. `2017 rx burning_topos.pdf` and `BFRSWallMap2017.pdf` Pictures of the units burned.
6. `Filters vs forest plot number.xlsx`: A more explicit listing of which forest unit each filter was collected at.

## Hints and possible approaches

+++

## Example Model

### Loading in Data
Let's start by uploading the data. We'll start with BlodgettCombinedBlobTable.csv. 

#### BlodgettCombinedBlobTable.csv

```{code-cell} ipython3
import pandas as pd

# define column names
col_names = ["Unused tags 1", "BlobID1", "Unused tags 2", 
            "1D Retention Time (min)", "2D Retention Time (sec)", 
            "Peak Height", "Peak Volume", "Peak volume/nearest internal standard peak volume", 
            "Calculated d-alkane retention index", "matched retention index", 
            "Unused tags 3", "Unused tags 4", "Unused tags 5", 
            "BlobID_2", "Filter number", "Unused tags 6", 
            "Mass concentration of compound (ng/m3)"]

# import csv file
df_blobtable = pd.read_csv("data/BlodgettCombinedBlobTable.csv", names=col_names)

df_blobtable
```

We can remove all of the columns with the unused tags and drop the NaN's.

```{code-cell} ipython3
unusedtags = ["Unused tags 1", "Unused tags 2", "Unused tags 3", 
                "Unused tags 4", "Unused tags 5", "Unused tags 6"]

#import numpy as np 

#df_blobtable.replace(np.inf, np.nan, inplace=True)

pd.set_option('use_inf_as_na',True)

df_blobtable = df_blobtable.drop(labels=unusedtags, axis=1)
df_blobtable = df_blobtable.dropna()
```

```{code-cell} ipython3
df_blobtable
```

#### All_ShrubCovOnly_01_16.xlsx

We can also load in the data from All_ShrubCovOnly_01_16.xlsx to get information about what plants are present at certain sites.

```{code-cell} ipython3
df_shrub = pd.read_excel("data/All_ShrubCovOnly_01_16.xlsx", sheet_name="16")

df_shrub
```

#### BFRSPlantCodes.xlsx

Now the data from BFRSPlantCodes.xlsx is read in to get information that links the shorthand code name to the real plant name.

```{code-cell} ipython3
df_plantnames = pd.read_excel("data/BFRSPlantCodes.xlsx")

df_plantnames
```

#### Run_Log.xlsx

We can also load in the data from Run_Log.xlsx for data collected during the flight and ground collections.

```{code-cell} ipython3
df_filter_flight = pd.read_excel("data/Run_Log.xlsx", sheet_name="Flight Log")

df_filter_flight
```

```{code-cell} ipython3
df_filter_ground = pd.read_excel("data/Run_Log.xlsx", sheet_name="Ground Station")

df_filter_ground
```

### Predicting Mass Concentration from BloblID

Now that some of the data has been read in, we can start to make our model. For this simple example model, we will try and predict a correlation between the BlobID, or the compound, and the amount of that compound in the smoke, given by the mass concentration. We will use RandomForestRegressor as part of sklearn. 

We start with splittig our data into a train/test split.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df_blobtable["BlobID1"], df_blobtable["Mass concentration of compound (ng/m3)"])
```

We will now fit the correlation between the BlobID and mass flow of that compound in the smoke. We will then test it on the test data.

```{code-cell} ipython3
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train.values.reshape(-1, 1), Y_train.values.reshape(-1, 1))
```

```{code-cell} ipython3
import matplotlib.pyplot as plt 

plt.plot(X_train.values.reshape(-1, 1), Y_train.values.reshape(-1, 1), '.')
plt.plot(X_test.values.reshape(-1, 1), Y_test.values.reshape(-1, 1), '.')
plt.plot(X_test.values.reshape(-1, 1), model.predict(X_test.values.reshape(-1, 1)), '.')
plt.xlabel('BlobID')
plt.ylabel('Mass Concentration of Compound in Smoke (ng/m3) ')
plt.legend(['Train Data', 'Test Data', 'Prediction']);
```

This model is not too conclusive about the mass concentration of the compound based on it's BlobID. This could be due to the data being collected at multiple plots and both measured on the ground and in the air. To improve upon this, you can try and see if there is a correlation between the mass concentration of a compound at a certain plot, or the elevation during a burn. 

There are also several other paths you can take for your project. However, this simple model does show how you can load in the data and start to use it.
