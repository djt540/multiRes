# multiRes

<div align="middle">
<picture>
  <img alt="NARMA on Rotor" src="https://github.com/djt540/images/blob/main/rot.png" width=33%>
</picture>
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://github.com/djt540/images/blob/main/multires-light.png">
  <img alt="multiRes Logo" src="https://github.com/djt540/images/blob/main/multires.png" width=15%>
</picture>
<picture>
  <img alt="NARMA on ESN" src="https://github.com/djt540/images/blob/main/esn%20NARMA10.png" width=33%>
</picture>
</div>

## Intro

multiRes is a simple implementation of space-time multiplexing of a reservoir computing system implemented fully in Python. This branch uses numpy for all matrix calculations. Example code can be found within `examples.py`, showing off multi node models including: a simple ESN, a rotor, a delay line, a rotor wrapping a delay line, and a delay line wrapping a rotor.

## Usage

multiRes has only been tested using Python 3.10. 
All required packages can be found in `requirements.txt` and installed using the command:
```Python
pip install -r requirements.txt
```

Documentation for each module (`nodes` and `util`) can be found inside their respective directories.

## Examples.py

This should be a basic introduction on usage for multiESN projects, showing:
+ Model Creation
+ Model Running
+ Model Parameter Optimisation using simulated annealing

This however is only the basics of what could be done with package, and does not show how to subclass the Node class - however this should be relatively obvious.
