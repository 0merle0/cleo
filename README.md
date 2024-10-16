# cleo
#### designing combinatorial libraries for protein optimization

this code base is split into basic elements of iterative optimization

### Setup
To set up the environment, use the following command guide
```
conda create --name cleo  python=3.9
conda activate cleo
pip install -r requirements.txt
```

### Launch surrogate training
To launch a training run, you can build a custom configs file and use the following
```
python train_surrogate.py -cn train_surrogate
```


![alt text](./figs/poster.jpg)
