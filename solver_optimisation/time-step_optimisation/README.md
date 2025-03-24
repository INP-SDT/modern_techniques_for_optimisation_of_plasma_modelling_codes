# Reinforcement learning time-stepping

This example implements a reinforcement learning method implemented in PyTorch to optimize time-step size control. It focuses on maximizing cumulative rewards over time, with the loss function directing the updates of network parameters to favour actions that lead to greater rewards, ultimately resulting in higher time steps. 

The input parameters include the number densities of the species being considered (in this case, electrons and ions) and the electric field. This simple example file serves as a prototype for implementing the procedure in other Python-based plasma modelling codes.
