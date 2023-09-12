# util package

***This readme was generated based on the docstrings from the code and has not been fully checked.***

## Submodules

## util.DataAnalysis module

<!-- !! processed by numpydoc !! -->

## util.ParamOptim module

<!-- !! processed by numpydoc !! -->

### *class* util.ParamOptim.ParamOpt(model: [Model](nodes.md#nodes.Model.Model), signal)

Bases: `object`

<!-- !! processed by numpydoc !! -->

#### *class* Param(instance: object, name: str, min: float = 0.1, max: float = 0.9, step: float = 0.05, cur_val: float = 0.7931736024319477, best_val: float = 0.7931736024319477)

Bases: `object`

Class for parameter optimization options.

* **Attributes:**
  **instance: object**
  : Instance of the object that contains the parameters to be optimised.

  **name: str**
  : Parameters name.

  **min: float = 0.1**
  : Bounding starting minimum value.

  **max: float = 0.9**
  : Bounding starting maximum value.

  **step: float = 0.05**
  : Value to be stepped per iteration.

  **cur_val: float = uniform(min, max)**
  : Current values to be tested.

  **best_val: float = cur_val**
  : Current best found values to minimize error.

<!-- !! processed by numpydoc !! -->

#### best_val*: float* *= 0.7931736024319477*

#### cur_val*: float* *= 0.7931736024319477*

#### instance*: object*

#### max*: float* *= 0.9*

#### min*: float* *= 0.1*

#### name*: str*

#### step*: float* *= 0.05*

#### anneal(params_list: list[[Param](#util.ParamOptim.ParamOpt.Param)], iterations: int = 100, initial_temp=150, verbose: bool = False)

Simulated Annealing for model hyperparameter optimisation.

Tests multiple parameter options based on their min, max, and step values.
Sets parameters to the best found during the algorithm.

This uses a simple multiplicative monotonic cooling schedule.

* **Parameters:**
  **params_list**
  : List of parameters to be optimised.

  **iterations**
  : Number of iterations for annealing to perform.

  **initial_temp**
  : Temperature bias in simulated annealing algorithm

  **verbose**
  : When true allows printing of error and iteration when best error is found.

  **Returns****——-****float**
  : Returns best error found

<!-- !! processed by numpydoc !! -->

#### grid_search()

Simple grid search implementation.

<!-- !! processed by numpydoc !! -->

#### *static* params_step(params_list: list[[Param](#util.ParamOptim.ParamOpt.Param)])

Steps the parameters for Simulated Annealing.

Each step take the current best value and adds the product of a value inside the bounds
and the step amount.

* **Parameters:**
  **params_list**
  : List of parameters to be stepped.

<!-- !! processed by numpydoc !! -->

#### *property* signal

<!-- !! processed by numpydoc !! -->

#### split_results(signal, splits=None)

Utility to split the output states

* **Parameters:**
  **signal**
  : input signal for the model to run on

  **splits**
  : size of state to split, default = [500, 4250, 4750]
* **Returns:**
  list[ndarray[Any, dtype[_SCT]]]
  : split states

<!-- !! processed by numpydoc !! -->

### util.ParamOptim.random(/)

<!-- !! processed by numpydoc !! -->

## util.test_ParamOptim module

<!-- !! processed by numpydoc !! -->

### util.test_ParamOptim.basic_po()

<!-- !! processed by numpydoc !! -->

### util.test_ParamOptim.test_anneal()

<!-- !! processed by numpydoc !! -->

### util.test_ParamOptim.test_params_step()

<!-- !! processed by numpydoc !! -->

## Module contents

<!-- !! processed by numpydoc !! -->
