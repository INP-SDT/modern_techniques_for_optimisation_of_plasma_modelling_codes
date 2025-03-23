import torch
from agent import Agent
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Number of input: electron density, ion density, electric field
n_input = 3
# Number of output: Probability of choosing a certain time step from 10 possible values
n_output = 10
# Create the agent
agent = Agent(n_input, n_output)
# Load trained model
model_path = 'model/model_4.pth'
agent.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

# Run the simulation
#### This should be inside the simulation loop ####
# Data structure for simulation state
# Example using random 1D arrays
ne = np.random.rand(200)
ni = np.random.rand(200)
ef = np.random.rand(200)
simulation_state = {
    "n_e": ne,
    "n_i": ni,
    "efeld": ef
}

# get old state
state_old = agent.get_state(simulation_state)
# get predicted dt
pred_dt, action = agent.get_action(state_old)
print("predicted time step size:", pred_dt)
###################################################
