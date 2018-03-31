# AutoencoderOptimizer

you need
python3 
tensorflow v1.5+
pandas



This project generates synthetic minority class samples using a trained  autencoder and Particle Sawarm optimization (PSO) algorithm

## Getting Started

These instructions will get you a copy of the project up and running on your local machine 
### Prerequisites

To use the project you need

```
* python3
* tensorflow v1.4+
* pandas
* numpy
```

### Installing

Clone or Download the project and use one of the following commands

* To train a new model
```
python main.py --dataset="THCA" --train=True --generate=False --epochs=4 n_layers=1000,500
```
* To generate representations using a previously trained  (test) model
```
python main.py --dataset="THCA" --train=False --generate= False
```
* To generate synthetic representations
```
python main.py --dataset="THCA" --train=False --generate= False
```



