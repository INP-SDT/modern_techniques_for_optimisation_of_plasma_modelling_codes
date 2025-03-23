# Conventional preconditioners applied to speed-up the GEC reference cell simulation

Before relying on developing new type of preconditioners, we want to test the how conventional ones work when applied for plasma modelling problems. Here, we test the performance of various preconditioners available
in PETSc by modelling RF discharge within the GEC reerence cell in argon at pressure of 100 mTorr and applied sinusoidal voltage. The voltage amplitude is 100 V (200 V peak-to-peak) without applied bias voltage. The model is implemented in FEDM.
