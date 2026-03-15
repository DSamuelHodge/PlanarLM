"""
constants.py — Default hyperparameters and model settings for PlanarLM.
Edit this file to change global defaults for experiments or reproducibility.
"""

# Model architecture
VOCAB_SIZE = 256
CHANNELS = 64
NUM_WIRES = 5
DEPTH = 10
ALPHA_INIT = 0.5
TIE_WEIGHTS = True

# Training/testing
BATCH = 4
SEQ_LEN = 256
