Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
rsync -a --no-g --no-o /home/dimaio/RoseTTAFold2/SE3Transformer/ $APPTAINER_ROOTFS/SE3Transformer/

%files
/etc/localtime
/etc/hosts
/archive/software/Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh

%post
# Switch shell to bash
rm /bin/sh; ln -s /bin/bash /bin/sh

# Common symlinks
ln -s /net/databases /databases
ln -s /net/software /software
ln -s /home /mnt/home
ln -s /projects /mnt/projects
ln -s /net /mnt/net

apt-get update
# required X libs
apt-get install -y libx11-6 libxau6 libxext6 libxrender1

# git
apt-get install -y git
apt-get clean

# Install conda
bash /opt/miniconda.sh -b -u -p /usr

# Install conda/pip packages
conda update conda

# Set python version before all else
conda install python=3.9

# Haven't tested if these need to be installed by themselves. Too much of a pain to check right now.
conda install \
   -c conda-forge \
   dm-tree=0.1.7 \
   pdbfixer=1.8.1 \
   mdtraj=1.9.7

# pytorch + dependancies
conda install \
   -c nvidia/label/cuda-12.1.0 \
   -c pytorch \
   -c pyg \
   -c dglteam/label/cu121 \
   -c anaconda \
   pip \
   ipython=8.8.0 \
   "ipykernel>=6.22.0" \
   numpy=1.22 \
   pandas=1.5.2 \
   seaborn=0.12.2 \
   matplotlib \
   jupyterlab=3.5.0 \
   pytorch==2.2 \
   pytorch-cuda==12.1 \
   dgl==2.0.0.cu121 \
   einops=0.7.0 \
   pyg


# open-babel needs to be conda installed last for reasons. Otherwise you get the error,
# Error while loading conda entry point: conda-libmamba-solver (libarchive.so.19: cannot open shared object file: No such file or directory)
# CondaValueError: You have chosen a non-default solver backend (libmamba) but it was not recognized. Choose one of: classic
conda install \
   -c conda-forge \
   openbabel=3.1.1 \
   
# pip extras
pip install \
   e3nn==0.5.1 \
   "hydra-core==1.3.1" \
   pyrsistent==0.19.3 \
   opt_einsum==3.3.0 \
   sympy==1.12 \
   omegaconf==2.3.0 \
   icecream==2.1.3 \
   wandb==0.13.10 \
   deepdiff==6.3.0 \
   assertpy==1.1 \
   biotite==0.36.1 \
   GPUtil==1.4.0 \
   addict==2.4.0 \
   fire==0.5.0 \
   tmtools==0.0.2 \
   plotly==5.16.1 \
   deepspeed==0.8.0 \
   biopython==1.80 \
   ipdb==0.13.11 \
   pytest==7.4.0 \
   openmm \
   colorlog \
   "ml-collections"

# Jax doesn't install nicely with other packages for reasons
pip install \
   jax==0.4.13

# Install git repos
pip install git+https://github.com/RalphMao/PyTimer.git
   
# SE3 transformer
pip install /SE3Transformer/

# Install apptainer so that this container can run other containers.
apt-get update
apt-get install -y software-properties-common
add-apt-repository -y ppa:apptainer/ppa
apt-get update
apt-get install -y apptainer

# Git needs to be installed separately for reasons.
# It also needs to be installed after installing apptainer, or errors will ensue.
#conda install git

# Clean up
conda clean -a -y
apt-get -y purge build-essential wget
apt-get -y autoremove
apt-get clean
rm /opt/miniconda.sh

%environment
export PATH=$PATH:/usr/local/cuda/bin

%runscript
/usr/bin/python "$@"

%help
Environment for running rf_diffusion_aa
