# Modern techniques for optimisation of plasma modelling codes

The low-temperature plasmas has a wide technological applications, spanning from chemical
processing and surface modification, over flow control, to biomedical applications. Therefore, there is need to understand underlying physical and chemical processes relevant to plasma production. A common way to gain a deeper insight into these processes is by numerical modelling and simulations using fluid, particle or hybrid models. For an accurate description of relevant processes in plasmas, one needs to consider many particle species and even more reactions occuring between them. This requirement results in necessity to solve a large system of partial differential equations (PDE) using numerical methods, for instance finite element method (FEM). The solution of the resulting system of equations takse days, weeks or even months. The recent development in the field of machine learning (ML) provides a great basis for the combination of conventional numerical methods with modern ML approach with the goal to optimise the solution procedure. The project aims to optimise the solution procedure of PDEs relevant to plasma modelling in terms of calculation time and thus facilitating the simulation of complex plasma systems.

## Development of preconditioners for iterative linear solvers

The set of nonlinear (often stiff) partial differential equations (PDEs) in the plasma model leads to a set of nonlinear equations after discretisation. These equations are solved using a Newton-based method, which involves solving a linear system of equations during every Newton iteration. To shorten the solution time and reduce memory usage, a linear system of equations can be solved using an iterative linear solver instead of a direct one. The iterative solvers have a high potential for parallelisation, allowing significant speed-up of the solution process on computing clusters. However, iterative solvers require a proper choice of preconditioners, which has a substantial impact on their performance. The goal is to design a preconditioner based on approximate inverse of the original matrix and test the performance gain. The examples of the use of preconditioners to improve the code performance are given [here](https://github.com/INP-SDT/modern_techniques_for_optimisation_of_plasma_modelling_codes/tree/main/tailoring_preconditioners).

## ML-based optimisation of mesh generation

Solving the PDEs using conventional numerical methods, such as FEM, requires spatial discretisation of the simulation domain. In case of discharges with steep gradients, such as streamers or in sheath regions of the discharge, mesh refinement is necessary. Additionally, the plasma sources often have complex designs, so difficulties in creating the required simulation domain geometry and mesh might arise. This task aims to develop ML-assisted mesh generation based on the discharge photos and the technical drawings of the plasma reactors. The goal is to generate and optimal mesh for the desired problem and specified accuracy with minimal time and effort.

## Optimisation of numerical solvers for PDEs

The nonlinear and iterative linear solvers, as well as time-stepping procedures, have a variaty of the parameters that can be optimised during the calculations. The goal is to use ML-based techiques (such as reinforcement learning) to automatically adjust these parameters along the course of calculations.

## ML-based solution of PDEs

Besides conventional methods for the solution of PDEs, the use of ML methods to solve them become quite popular recently. The goal is to extend available methods and extend them to certain types of PDEs.

## Acknowledgment
The project is supported by DAAD Programm des Projektbezogenen Personenaustauschs (PPP) mit Serbien - project number 57703239.

## Contact

[aleksandar.jovanovic@inp-greifswald.de](mailto:aleksandar.jovanovic@inp-greifswald.de)

[markus.becker@inp-greifswald.de](mailto:markus.becker@inp-greifswald.de)
