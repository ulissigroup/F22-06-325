FROM jupyter/datascience-notebook

RUN mamba install --quiet --yes \
    jupyterlab_code_formatter black yapf isort autopep8 jupytext nbgitpuller jupyter-book python-lsp-server jupyterlab-lsp pre-commit && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN pip install --quiet --no-cache-dir jupyterlab_tabnine && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN jupyter labextension install jupyterlab-jupytext


