# cleo
#### designing combinatorial libraries for protein optimization

this code base is split into basic elements of iterative optimization

### Setup
To set up the environment, use the following command guide:
```
conda create --name cleo  python=3.9
conda activate cleo
pip install -r requirements.txt
```

### Launch surrogate training
To launch a training run, you can build a custom configs file and use the following:
```
python train_surrogate.py -cn train_surrogate
```

### General workflow
To contribute to the codebase please follow the general workflow:
1. create a new branch
2. develop and push changes to new branch
3. make a pull request to merge your branch into main

Make sure you merge the latest changes from main before submitting your PR.

![alt text](./figs/poster.jpg)
