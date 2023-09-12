# nodes package

***This readme was generated based on the docstrings from the code and has not been fully checked.***

## Submodules

## nodes.DelayLine module

<!-- !! processed by numpydoc !! -->

### *class* nodes.DelayLine.DelayLine(tau: int = 3, fb_str: float = 0.5, eta: float = 0.1)

Bases: [`Node`](#nodes.Model.Node)

Implements virtual nodes using a simple mask and time multiplexing.
Based on: “Information processing using a single dynamical node as complex system”
[SDN]

* **Parameters:**
  **tau**
  : Number of virtual nodes

  **fb_str**
  : Hyperparameter Feedback strength

  **eta**
  : Hyperparameter input scaling
* **Attributes:**
  **\_mask**
  : Mask the size of the virtual nodes applied to the signal in the forward function.
    <br/>
    This is not the mask used in the paper, output is highly dependent on the mask,
    however this is the best mask I could find for this task.

  **\_wrapped**
  : Node that the forward function passes the signal to.

  **\_v_states**
  : Internal states of the virtual nodes, only contains one timestep before overwritten

<!-- !! processed by numpydoc !! -->

#### *property* fb_str

* **Returns:**
  Feedback strength for the delay line

<!-- !! processed by numpydoc !! -->

#### forward(signal: ndarray)

Forward function of the delay line. Per timestep a mask is applied to the incoming signal and added to the
product of the previous state and feedback strength. For the number of virtual nodes the modified signal is
through a dynamical node (the wrapped nodes forward function). The final state from v_states is returned.

* **Parameters:**
  **signal**
  : Input signal for the delay line node.

  **Raises****——****Exception**
  : Delay Line has nothing to wrap if wrapped = None

  **Returns****——-****np.ndarray**
  : Returns output states (the delay lines final node output) from each node in the chain

<!-- !! processed by numpydoc !! -->

#### *property* tau

* **Returns:**
  Number of virtual nodes in the delay line

<!-- !! processed by numpydoc !! -->

#### *property* wrapped*: [Node](#nodes.Model.Node)*

Node that the forward function passes the signal to.
Returns
——-
The node that this node currently wraps in the model

<!-- !! processed by numpydoc !! -->

## nodes.Model module

<div align="middle">
<img src="https://github.com/djt540/images/blob/main/node.png" width=35%>
</div>

<!-- !! processed by numpydoc !! -->

### *class* nodes.Model.Model(node_list: tuple[[Node](#nodes.Model.Node), ...])

Bases: `object`

The Model class handles constructing and running models based on a list of nodes.

The model class will run through the list of nodes and string their ‘wrapped’ parameter
allowing their forward functions to chain together into a completed system.

The run method will run the constructed model using a given signal in, storing and returning
the output states (stored as states in the model object). These states are later used
for ridge regression learning for the model.

### Notes

NARMA is the test I was using for all the models I created - so there is a NARMA
function inside the model class, however this should really be in its own set of
utility files along with the error calculations and plotting functions.

Error calculation here means normalised root-mean-square error, as seen in the following
formula:

$$\sqrt{\frac{1}{M}\frac{\sum_{k=1}^{M}(\hat{y_k}-y_k)^2}{\sigma ^2(y_k)}}$$


found in the supplementary information from the single dynamical node paper [SDN]

The error calc function call the ridge regression function as well based on the
states and target provided.

* **Parameters:**
  **node_list**
  : List of nodes to be made into a model. The first node in the tuple will be the first
    node to receive the signal, with the last one receiving the signal last.
* **Raises:**
  IndexError
  : If there are not enough nodes in the list to create a model (minimum 2 nodes).

  AttributeError
  : If the last node doesnt have a num_nodes attribute, e.g. a delay line is the final node.

  AttributeError
  : If not all the nodes in the list have a name.

#### SEE ALSO
[`Node`](#nodes.Model.Node)
: Node class required to form a model.

* **Attributes:**
  **states**
  : Output of the model. Used for Ridge Regression.

  **gamma**
  : Parameter used in ridge regression.

  **node_list**
  : List of nodes in the model, in order.

  **model_len**
  : Number of node objects in the model (however this currently doesnt add the number of nodes
    hidden inside a NodeArray).

  **first_node**
  : First node in the model.

  **last_node**
  : Last node in the model.

  **num_nodes**
  : Number of nodes in the model (size of all spilt signals added together)
    e.g. a model with 3 ESN’s each 100 nodes in size will have a num nodes of 300,
    a model with 1 ESN with 100 nodes will have a num_nodes of 100.

  **node_names**
  : List of the nodes names, these are all set in the node classes init and
    required to create a model.

<!-- !! processed by numpydoc !! -->

#### *static* NARMAGen(signal)

<!-- !! processed by numpydoc !! -->

#### *static* NRMSE(pred, target)

<!-- !! processed by numpydoc !! -->

#### error_test(train, train_target, compare, compare_target)

<!-- !! processed by numpydoc !! -->

#### ridge_regression(states, target)

<!-- !! processed by numpydoc !! -->

#### run(signal: ndarray)

Run function for the model.

This handles the run loop for the model, which is the length of the signal provided.
An output of the signals length and the depth of the number of total nodes[#num_nodes]_ in the
model is created. The output is the state of the model for each timestep.

Run simply passes the signal to the forward function of the first node in the
node_list and would need to be modified if a more complex model class was created[#fut]_.

### Footnotes

in the model class documentation.

this is beyond the scope of this project.

* **Parameters:**
  **signal: np.ndarray**
  : Signal to run the model, for NARMA this will be an array of random noise.

  **Returns****——-****np.ndarray**
  : Output states.

<!-- !! processed by numpydoc !! -->

#### simple_plot(prediction, target)

<!-- !! processed by numpydoc !! -->

### *class* nodes.Model.Node

Bases: `ABC`

Abstract class for nodes to inherit from.

The node class template which enforces the use of a forward function
to allow chaining of future nodes in the line and for use of the
model class. Most classes here will require a wrapped parameter, however
it is not required.

#### SEE ALSO
[`Model`](#nodes.Model.Model)
: Model class - constructs and runs a model using Nodes

`DelayLine`
: A node subclass with wrapping functionality

### Notes

This allows chaining of nodes, with current implementations
only allowing simplistic chaining (one directional line of nodes, other
shapes have not been tested or properly implemented.)

### Methods

| **forward(self, signal: np.ndarray) -> np.ndarray**   |    |
|-------------------------------------------------------|----|
<!-- !! processed by numpydoc !! -->

#### *abstract* forward(signal: ndarray)

Abstract method for the Node classes forward function.
Parameters

Where modification of the incoming signal will take place
before the signal is passed onto the next node (if the current
node is wrapping another node).
———-
signal : np.ndarray

> Input signal for the node.

### Returns

np.ndarray
: Returns the output states

<!-- !! processed by numpydoc !! -->

## nodes.NodeArray module

<!-- !! processed by numpydoc !! -->

### *class* nodes.NodeArray.NodeArray(nodes: list[[Node](#nodes.Model.Node)])

Bases: [`Node`](#nodes.Model.Node)

Array of Node objects

* **Parameters:**
  **nodes**
  : List of the nodes to for the incoming signal to be passed to. (This is effectively a wrapped list.)

### Notes

If the objects in the array are reservoirs then the NodeArrays num_nodes equals number of nodes
in the reservoir x the number of objects.

If note the number of nodes equals the number of objects in the array.

* **Attributes:**
  **obj_nodes**
  : Number of nodes each object contains, currently can only support objects with equal number of nodes.

  **num_nodes: int**
  : Number of nodes total in the forward parts of the model, see notes for more information on conditional
    values of num_nodes.

<!-- !! processed by numpydoc !! -->

#### forward(signal)

Forward function for the node arrays, passes even slices of the input signal into the
nodes in the nodes list, currently this is equally sized slices.

* **Parameters:**
  **signal**
  : Input signal for the NodeArray node

  **Returns****——-****np.ndarray**
  : Output states of all the objects concatenated together.

<!-- !! processed by numpydoc !! -->

## nodes.Reservoir module

<!-- !! processed by numpydoc !! -->

### *class* nodes.Reservoir.InputMask(num_nodes, connectivity: float = 0.5)

Bases: [`Node`](#nodes.Model.Node)

Applies an input mask of size num_nodes with a uniform distribution of 1 and -1.

* **Parameters:**
  **num_nodes**
  : The number of nodes for the mask.
* **Attributes:**
  **w_in**
  : The mask that is applied to the incoming signal.

  **\_wrapped**
  : Node that the forward function passes the signal to.

<!-- !! processed by numpydoc !! -->

#### forward(signal)

Applies the input mask to the signal.

* **Parameters:**
  **signal**
  : Input signal for the delay line node.
* **Returns:**
  np.ndarray
  : Returns output states including changes from future nodes

<!-- !! processed by numpydoc !! -->

#### *property* wrapped*: [Node](#nodes.Model.Node)*

Node that the forward function passes the signal to.
Returns
——-
The node that this node currently wraps in the model

<!-- !! processed by numpydoc !! -->

### *class* nodes.Reservoir.Reservoir(num_nodes: int, connectivity: float = 0.1, leak: float = 0.75, in_scale: float = 1, spec_r: float = 0.95)

Bases: [`Node`](#nodes.Model.Node)

An implementation of an Echo State Network

* **Parameters:**
  **num_nodes**
  : The size of the reservoir - the number of nodes in the reservoir.
    Typically larger reservoirs have higher capacities.

  **connectivity**
  : The connectivity percentage of the reservoirs weight matrix, with
    each connection less than the connectivity being set to 0 in weight instantiation.

  **leak**
  : The rate at which previous state feedback affects the reservoir.

  **in_scale**
  : The rate at which the current feedback affects the state of the reservoir.

  **spec_r**
  : The spectral radius of the weight matrix of the reservoir.
* **Attributes:**
  **wrapped**
  : Node that the forward function passes the signal to.
    Note: currently goes nowhere, however I believe allowing DeepESNs should be easy
    with this framework.

  **prev_state**
  : State of the reservoir in its perceived last timestep (i.e. the last time a
    signal was passed in). This is used for the update/forward calculation.

  **w_res**
  : Internal weights of the reservoir.

<!-- !! processed by numpydoc !! -->

#### forward(signal)

Forward function of the reservoir.
: Following the following update equation:

  $$ x(n+1)= (1-leak) \cdot x_n + tanh(leak * x_n \times W_{res} + in \textunderscore scale * signal) $$


### Parameters

signal
: Input signal for the reservoir node

* **Returns:**
  Output states for the reservoir

<!-- !! processed by numpydoc !! -->

#### *property* in_scale

Scaling for the input into the reservoir
Returns
——-
float

> value of in_scale
<!-- !! processed by numpydoc !! -->

#### *property* leak

Leak rate for previous states
Returns
——-
Value of the Leak property

<!-- !! processed by numpydoc !! -->

#### reset_states()

<!-- !! processed by numpydoc !! -->

## nodes.Rotor module

<!-- !! processed by numpydoc !! -->

### *class* nodes.Rotor.Rotor(rot_num: int, tot_inputs: int)

Bases: [`Node`](#nodes.Model.Node)

Rotates inputs signal and output states
Based on: Rotating neurons for all-analog implementation of cyclic reservoir computing [RNR]

* **Parameters:**
  **rot_num**
  : Number of Nodes in the system

  **tot_inputs**
  : Total number of inputs in the following node array

### Notes

*This Node is aimed to work with the NodeArray node, where Rotor rotates the
incoming signals and outgoing states the NodeArray passes these signals onto
multiple Nodes, splitting the signal evenly between the nodes.*

<!-- !! processed by numpydoc !! -->

#### forward(signal)

Takes in a signal of size n where n is number of inputs per object x number
of objects. So for 3 ESNs with 100 inputs each rotation will be of size 100.
The rotation amount is the product of the rotation amount and the current timestep.

* **Parameters:**
  **signal**
  : Input signal for the rotor node.

  **Returns****——-****np.ndarray**
  : Output States

<!-- !! processed by numpydoc !! -->

#### *property* wrapped*: [Node](#nodes.Model.Node)*

Node that the forward function passes the signal to.
Returns
——-
Node

> The node that this node currently wraps in the model
<!-- !! processed by numpydoc !! -->

## nodes.test_Model module

<!-- !! processed by numpydoc !! -->

### nodes.test_Model.test_error_test()

<!-- !! processed by numpydoc !! -->

## nodes.value_tester module

<!-- !! processed by numpydoc !! -->

### *class* nodes.value_tester.BlankLine(num_nodes, verbose=True)

Bases: [`Node`](#nodes.Model.Node)

<!-- !! processed by numpydoc !! -->

#### forward(signal)

Abstract method for the Node classes forward function.
Parameters

Where modification of the incoming signal will take place
before the signal is passed onto the next node (if the current
node is wrapping another node).
———-
signal : np.ndarray

> Input signal for the node.

### Returns

np.ndarray
: Returns the output states

<!-- !! processed by numpydoc !! -->

### nodes.value_tester.test_rotor()

<!-- !! processed by numpydoc !! -->

## Module contents

<!-- !! processed by numpydoc !! -->
