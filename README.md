# CUDA-Kernel-Node-Editor-For-Houdini

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

by Dongying Liu, Wenqing Wang, and Yuchia Shen.

# Overview
TODO: Dongying/Wenqing/Yuchia


# Code Generator Pipeline
## Creat Houdini VOP Node
TODO: Dongying
## VOP to JSON
TODO: Dongying/Wenqing
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

