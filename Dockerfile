FROM jupyter/datascience-notebook

RUN mamba install --quiet --yes \
    jupyterlab_code_formatter \
    black \
    yapf \
    isort \
    autopep8 \
    jupytext \
    nbgitpuller \
    jupyter-book \
    jupyterlab-myst \
    jupyterlab-spellchecker \
    git-lfs \
    lmfit \
    plotly \
    pre-commit && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

#    python-lsp-server \
#    jupyterlab-lsp \
    
RUN jupyter labextension install jupyterlab-jupytext


# OCP repo requirements
RUN echo '' > /opt/conda/conda-meta/pinned
RUN mamba uninstall nomkl --quiet --yes
RUN mamba install --quiet --yes \
    -c pyg -c pytorch -c conda-forge -c nvidia \
    numba \
    lmdb \
    ase \
    pip \
    pyyaml \
    tqdm \
    tensorboard \
    pandoc \
    nbsphinx \
    sphinx \
    black \
    wandb \
    lmdb \
    submitit \
    pytest \
    python-lmdb \
    pyg \
    'pytorch=1.11' \
    torchvision \
    torchaudio \
    'python=3.9' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

ENV OMP_NUM_THREADS=1 

USER root
RUN git clone https://github.com/Open-Catalyst-Project/ocp /opt/ocp && fix-permissions /opt/ocp
USER $NB_UID

RUN cd /opt/ocp && python setup.py develop

# Add texlive-science for mhchem package!
USER root
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    texlive-science && \
    apt-get clean && rm -rf /var/lib/apt/lists/*USER $NB_UID
USER $NB_UID
