# Software Environments

Like in 06-262, we will mostly use python in this class. In addition to google colab, I have also set up a course JupyterHub server.

## Course JupyterHub

As a trial this semester, I am running a JupyterHub server for this course to provide persistent python environments with a number of helpful python packages pre-installed. This has a number of advantages:
* The data is persistent. If you write or download files in the environment, they will be there the next day/week/month.
* You can use other languages besides python.
* You have access to a bash terminal (helpful for learning unix/bash commands, moving files around, etc).
* You can ask for my (or the TAs) help, and we can collaboratively edit and interact with your notebooks.

### Access to Jupyterhub server (https://laikapack-head.cheme.cmu.edu/hub):
* You must be on CMU campus, or use the CMU VPN
* You need to make a github account (github.com) and email Prof. Ulissi your githubid (e.g. Prof. Ulissi's github profile is at https://github.com/zulissi, so his username is `zulissi`). This is only used for authenatication/logging in to the server. 
* Access your jupyterlab environment at https://laikapack-head.cheme.cmu.edu/hub
* Use the minimum resources your need (if things are running fast, stick with the smallest server). If you need more resources, go to file, Hub Control Panel, stop your server, and start one with more resources. For
* Every user is given 20gb of storage space that should be more than enough for the course. The file system is backed up daily in case you accidentally delete things. You can see recent backups of each folder by going to the `.snap` directory within any folder. 

Be aware that anything you store on the jupyterhub server Prof. Ulissi has access to if needed. Normal CMU computing policies on allowed used of resources apply (no pirating, illegal content, etc etc). Treat this as you would a lab computer in Doherty's lab.

This is an experiment and run on Prof. Ulissi's research cluster, so please be patient if there are hiccups and report anything strange on piazza.

## Google Colab

This should work like before. If you're on a notebook page, there should be a little "open in google colab" button if you mouse over the rocket symbol. You can also download the ipynb files from the course github page and upload them to colab yourself. Remember that local files you write (outside of the notebook) are not saved in colab!

## Other software/environments

If you have strong feelings about the language you want to use, or using local vs cloud environments (e.g. something fancy setup and installed on your own laptop/desktop instead of the cloud environments provided), you are welcome to do so. However, you will mostly be on your own if you have compatibility problems/etc. 

**Windows:** I would suggest running python/etc in WSL (Windows Subsystem for Linux) and use mamba for package installation in python. If you want more details, ask and I will update this section. This is what I use for day-to-day work and research.

**Macs:** I don't have much experience with OSX, but it should be possible to install mamba and most python packages. If you want more details, ask and I will update this section.

**Linux:** I would suggest you install mamba and use that for python package installation. Check with the TAs or make a github issue and I can fill in details here if needed.
