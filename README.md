# CUDA-Kernel-Node-Editor-For-Houdini

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

by Dongying Liu, Wenqing Wang, and Yuchia Shen.

# Project Overview
## **Goals**
Provide an cuda kernel node editor for Houdini. 
Users can create various types of effects based on Houdini VOP node, and our tool can auto-generate both CPU (for debugging) and CUDA code based on the VOP node.

## **Target Users** 
Technical Artist
Even if the technical artist or other users who is not familiar with those holy graphics APIs can do CUDA development with our tool.
The only thing our user need to worry about is how to create effect with Houdini VOP node. It doesn't matter to them how the algorithem will actually generate backend.

## **Why Houdini**
Among all the cool game engines which also has node base editor like Unreal and Unity, why we choose Houdini?

```Visual Debugging System``` 
For our target users, Houdini has a powerful visual debug system, which means we can immediatly see the result from the scene view window whenever we make changes. And this perticularly benifit our users, who will create and visualize the effects with Houdini VOP Node.
Except the realtime result. Houdini has this spreadsheet where we can check attributes for every points primitive of the geometry like position, color, normal and even more custom attributes to see if the value reaches our expectations.

```Powerful Python API``` 
For us the code generator development, Houdini has a powerful python API. We can write python with python nodes in Houdini and see real time result with our code wrote. And what's more, with the print node of the VOP node, we can easily generate ground truth for our code generator to debug for our code generator.

## **Is This Project Highly Dependence on Houdini?**
The answer is... No!
We designed our code generator decoupling with JSON, which means every steps of our code generator will communicate with their next step with JSON.
Houdini is only a front end application. as long as the application can generate the same formatted JSON(which the format is designed by us), we can use the follow up python program to generate code immediatly. For more information, please checkout the next part: Code Generator Pipeline.

## **What's next? Future Potential?**
Now we are only generating CUDA code base on the information we get from the JSON files. With all the information needed stores in the JSON files, we can generate code for multiple platform later.

# Code Generator Pipeline
## Creat Houdini VOP Node
TODO: Dongying
## VOP to JSON
The outputs of this stage are a series of JSON files.

The first step in this phase is to work with the developer in the next phase to determine the format of the JSON file and the required information based on the reference code (here we refer to Houdini's VEX code). For example, in order to parse the `add` node in our VOP network into the following line of code, the information that must be included in the JSON is: node name, operation, input list, output list, as well as the connections to inputs and outpus.
```
float sum = input1 + input2;
```
## JSON to Code

The JSON to Code phase has two main inputs. One is the JSON file from the previous phase, which contains information about each node in the VOP network and their connections. The other is a template file that defines the main components of the target code snippet. The final output of this stage is all the files we need to generate the solution, including the CPU/GPU header/source files and our generic code files.

<img height = "400" alt="code_generate_pipeline" src="img/code_generate_pipeline.png">

### CPU-GPU Generic Code Generation
#### Concept Introduction
CPU-GPU Generic Code is a code segment that can be run on both CPU and GPU. In the generated target files, the generic code contains all the functions used to implement the core algorithm and will be called by both CPU and GPU code in the later stage.
#### Simple example
Smoke particle effect would be a good example here to help understand what is generic code. The core algorithm of this effect has only one line of code:
```
pos += v*dt
```
Now, suppose we need to implement this effect on both the CPU and GPU.
<img height = "200" alt="code_generate_pipeline" src="img/generic_code_example.png">

On the CPU, we will use a for loop to iterate over each particle and update their position using this formula. While on the GPU side, we'll call kernel launch and let each kernel compute the new position in parallel.

Although the way of traversing the particles is different, the code used to calculate its new position is the same. This line of code is an example of CPU-GPU Generic code.




### CPU Code Generation

### CUDA Code Generation


## Read in Houdini data (CGBuffer)
TODO: Yuchia
## Code to OBJ
TODO: Yuchia
## Test OBJ Back in Houdini
TODO: Dongying

# Results
## Example 1: Simple particle
TODO: 
1) Add visualization results (compare Houdini/CPU/GPU created effects in Houdini) - Dongying
2) Add generated files overview - Wenqing
3) Add brief description - Wenqing

### Particle Emitter
## Example 2: Tornado (velocity field)
Todo: 
1) Add final visualization results 
### Particle Emitter
TODO: 
1) Add brief description - Yuchia
2) Add Houdini visualization result - Dongying

