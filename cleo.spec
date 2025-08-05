Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%files
/etc/localtime
/etc/hosts
/etc/apt/sources.list
/archive/software/Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh

%post
## Switch shell to bash
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
#  git
apt-get install -y git
apt-get install -y libaio-dev
apt-get clean

# Install conda
bash /opt/miniconda.sh -b -u -p /usr

# install everything
# lock python version to 3.10.8
#      cuda version to 12.4
#      torch version to 2.2
conda install \
   -c conda-forge \
   -c nvidia/label/cuda-12.4.0 \
   -c pytorch \
   -c pyg \
   -c dglteam/label/th24_cu124 \
   -c https://conda.rosettacommons.org \
   python==3.11 \
   openbabel \
   pip \
   ipython \
   ipykernel \
   numpy \
   pandas \
   seaborn \
   matplotlib \
   jupyterlab \
   pyrosetta \
   pyg \
   dgl==2.4.0.th24.cu124 \
   pytorch==2.4 \
   pytorch-cuda==12.4 \
   cuda

# pip extras
pip install e3nn \
   omegaconf \
   hydra-core \
   pyrsistent \
   opt_einsum \
   sympy \
   omegaconf \
   icecream \
   wandb \
   deepdiff \
   assertpy \
   pytest \
   pytest-benchmark \
   einops \
   submitit \
   biopython \
   botorch \
   rich \
   pytorch-lightning

# deepspeed
pip install deepspeed


# Clean up
conda clean -a -y
apt-get -y purge build-essential git wget
apt-get -y autoremove
apt-get clean
#rm /opt/miniconda.sh

%environment
export PATH=$PATH:/usr/local/cuda/bin

%runscript

exec python "$@"