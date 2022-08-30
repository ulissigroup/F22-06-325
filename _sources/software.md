# Software Environments

Like in 06-262, we will mostly use python in this class. In addition to google colab, I have also set up a course JupyterHub server.

## Printing your assignments

As you saw in 06-262, printing jupyter notebooks in colab is a bit hit or miss and sometimes the output is a hard to read or missing graphs. The situation should be better with jupyterlab:
* Try file -> save and export notebook as -> pdf
* Try file -> save and export notebook as -> html, open the link, then save the output as PDF

Either way, double check the resulting PDF is complete and contains all of the output!

## Course JupyterHub

As a trial this semester, I am running a JupyterHub server for this course to provide persistent python environments with a number of helpful python packages pre-installed. This has a number of advantages:
* The data is persistent. If you write or download files in the environment, they will be there the next day/week/month.
* You can use other languages besides python.
* You have access to a bash terminal (helpful for learning unix/bash commands, moving files around, etc).
* You can ask for my (or the TAs) help, and we can collaboratively edit and interact with your notebooks.

### [Access to JupyterHub server](https://laikapack-head.cheme.cmu.edu/hub)
* You must be on CMU campus, or use the [CMU VPN](https://www.cmu.edu/computing/services/endpoint/network-access/vpn/)
* Access your jupyterlab environment at https://laikapack-head.cheme.cmu.edu/hub
* Use the minimum resources your need 
    * if things are running fast, stick with the smallest server). If you need more resources, go to file, Hub Control Panel, stop your server, and start one with more resources
* Every user is given 10gb of storage space that should be more than enough for the course. The file system is backed up daily in case you accidentally delete things. You can see recent backups of each folder by going to the `.snap` directory within any folder. If you happen to fill the 10gb of space, email Prof Ulissi and he will raise your limit.

Be aware that anything you store on the jupyterhub server Prof. Ulissi and the TAs have access to if needed. Normal CMU computing policies on allowed used of resources apply (no pirating, illegal content, etc etc). Treat this as you would a lab computer in Doherty lab.

This is an experiment and run on Prof. Ulissi's research cluster, so please be patient if there are hiccups and report anything strange on piazza.

### JupyterHub Usage Tips/Tricks

A few suggestions if you're having issues with JupyterHub (I'll update this throughout the semester)
* **Trouble saving a notebook** If your notebook doesn't seem to be saving for some reason, you can copy and paste all cells into a new notebook and save that. Edit -> select all cells, Edit -> copy cells, then go to a new notebook and Edit -> Paste all cells below. You can also try "save as" for the notebook and choose a new filename. If this is a recurring issue post on piazza so we can see if there's an underlying issue.
* **HW editing** Before working on a homework, I would suggest duplicating the homework assignment (right click the file, then click duplicate), then edit that. This will ensure the homework doesn't get overwritten if I make changes to the homework and github resyncs. 

## Google Colab

This should work like before. If you're on a notebook page, there should be a little "open in google colab" button if you mouse over the rocket symbol. You can also download the ipynb files from the course github page and upload them to colab yourself. Remember that local files you write (outside of the notebook) are not saved in colab!

## Other software/environments

If you have strong feelings about the language you want to use, or using local vs cloud environments (e.g. something fancy setup and installed on your own laptop/desktop instead of the cloud environments provided), you are welcome to do so. However, you will mostly be on your own if you have compatibility problems/etc. 

If you have suggestions for what's worked for you, submit a pull request to the github repo with your suggestions (or I can show you how!)

**All:** Follow the directions below to install a working unix environment, then use the following:
* Inside a terminal install mambaforge: https://github.com/conda-forge/miniforge#install
* Close and open your terminal after installation to make sure the conda scripts are used for activate/etc
* Create an environment with
```
mamba create -n myjlabenv jupyterlab -c conda-forge
conda activate myjlabenv  # activate our environment
jupyter lab               # this will start up jupyter lab and open a browser
```
* Install additional packages (numpy, scipy, seaborn, etc)
``` 
conda activate myjlabenv
mamba install -c conda-forge scipy seaborn numpy matplotlib
```
You can see what I installed by looking at https://github.com/ulissigroup/F22-06-325/blob/main/Dockerfile#L3 and everything listed in https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-datascience-notebook


**Windows:** I would suggest running python/etc in WSL (Windows Subsystem for Linux) and use mamba for package installation in python.
* Install WSL using the directions here: https://docs.microsoft.com/en-us/windows/wsl/install

**Macs:** I don't have much experience with OSX, but it should be possible to install mamba and most python packages. If you want more details, ask and I will update this section. 

**Linux:** I would suggest you install mamba and use that for python package installation. Check with the TAs or make a github issue and I can fill in details here if needed.
