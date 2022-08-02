FROM jupyter/datascience-notebook

RUN mamba install --quiet --yes \
    jupyterlab_code_formatter black yapf isort autopep8 && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN pip install --quiet --no-cache-dir jupyterlab && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
