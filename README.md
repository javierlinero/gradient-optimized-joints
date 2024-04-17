<a name="readme-top"></a>


<br />
<div align="center">
  <a href="https://github.com/javierlinero/gradient-optimized-joints">
    <img src="/src/images/logo.png" alt="Logo" width="100" height="100">
  </a>
  <h3 align="center">Gradient-Based Stiffness Optimization</h3>

</div>

## About The Project:
This project focuses on enhancing the design of woodworking joints through gradient-based stiffness optimization. Traditional joint methods often face issues with durability and environmental impact. Utilizing the Finite Element Method (FEM) and gradient descent techniques, this project aims to optimize the shape and efficiency of interlocking joints, such as dovetails and lap joints. By improving joint stiffness, the structural integrity and longevity of assembled objects are significantly enhanced, reducing material costs and simplifying assembly processes. This approach not only broadens the application possibilities of complex joinery but also contributes to more sustainable manufacturing practices.

## Main Features:

### Shapes.py
**What it does**: The shapes.py module provides functionalities to create and manipulate the shapes required for finite element analysis, using pygmsh to generate triangular mesh geometries for various woodworking joint designs.
### FEM.py
**What it does**: The fem.py module handles the setup and execution of finite element analyses for optimizing woodworking joint designs, utilizing mesh generation and stiffness evaluations to iteratively refine joint geometry.
### optimizer.py
**What it does**: The optimizer.py module orchestrates the optimization of woodworking joint designs by applying finite element methods and gradient-based optimization techniques. It integrates penalty structures and regularizers to enhance joint configurations for better mechanical performance, ensuring convergence through rigorous iterative adjustments.
### init_param_script.py
**What it does**: 
The init_param_script.py script randomizes parameters for different woodworking joints to ensure varied configurations and avoid local minima.

## Prerequisites
For a detailed guide on setting up a local version of this project, please refer to our comprehensive tutorial [here](/src/prerequisites/README.md).


### Built With:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

### Built on:
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)