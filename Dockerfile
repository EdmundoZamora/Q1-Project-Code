# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
# ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2021.2-stable

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
# ARG BASE_CONTAINER=ucsdets/datascience-notebook:2021.2-stable

# scipy/machine learning (tensorflow, pytorch)
#https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2021.3-42158c8

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root
# viewing images doesnt work yet.
#RUN docker run -it --user root-- sudo apt-get install -y feh
# RUN sudo apt-get update -y
# RUN apt-get -y install feh
# RUN apt-get -y install X

# 3) install packages using notebook user
USER jovyan 
# RUN conda install -y environment dependencies seen on environment.yaml
RUN pip install --no-cache-dir pandas numpy vak scipy torch librosa sox multipledispatch 
# clone the repository and switch working directory to run the methodology.
# make repo directory
#RUN mkdir -p ~/repos
# cd into repo directory
#WORKDIR ~/repos
# clone git repository
RUN git clone https://github.com/EdmundoZamora/Methodology5.git
# cd into Methodology directory in the repos directory.
WORKDIR Methodology5
#RUN conda env create -f tweety.yaml
#RUN conda activate tweety
# run command or entrypoint
# ENTRYPOINT [ "executable" ]
CMD [ "python","-u", ".\run.py", "data", "features", "model", "evaluate" ]
# spin the container and find the directory. Troubleshoot the run.py