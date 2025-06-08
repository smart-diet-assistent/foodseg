# Wandb configuration for FoodSeg LRASPP project
"""
Configuration file for Weights & Biases integration.
Set your wandb settings here before training.
"""

# Wandb project settings
WANDB_PROJECT = "foodseg-lraspp"
WANDB_ENTITY = None  # Set to your wandb username/organization if needed

# Logging settings
LOG_GRADIENTS = True  # Whether to log model gradients
LOG_PARAMETERS = True  # Whether to log model parameters
LOG_FREQUENCY = 100  # How often to log gradients/parameters (in steps)
BATCH_LOG_FREQUENCY = 50  # How often to log batch metrics (in batches)

# Artifact settings
SAVE_MODEL_ARTIFACTS = True  # Whether to save model checkpoints as artifacts
SAVE_PREDICTION_ARTIFACTS = True  # Whether to save prediction visualizations
SAVE_TRAINING_CURVES = True  # Whether to save training curves

# Run settings
WANDB_MODE = "online"  # "online", "offline", or "disabled"
WANDB_NOTES = None  # Additional notes for the run
WANDB_TAGS = ["semantic-segmentation", "lraspp", "mobilenetv3", "foodseg103"]

# Advanced settings
WATCH_MODEL = True  # Whether to use wandb.watch() on the model
WATCH_LOG = "all"  # What to log with wandb.watch(): "gradients", "parameters", "all", or None
WATCH_LOG_FREQ = LOG_FREQUENCY

# Environment variables (optional)
# Set these environment variables if you want to configure wandb externally:
# WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE, WANDB_NOTES, WANDB_TAGS

def get_wandb_config():
    """Get the complete wandb configuration."""
    return {
        "project": WANDB_PROJECT,
        "entity": WANDB_ENTITY,
        "mode": WANDB_MODE,
        "notes": WANDB_NOTES,
        "tags": WANDB_TAGS,
        "log_gradients": LOG_GRADIENTS,
        "log_parameters": LOG_PARAMETERS,
        "log_frequency": LOG_FREQUENCY,
        "batch_log_frequency": BATCH_LOG_FREQUENCY,
        "save_model_artifacts": SAVE_MODEL_ARTIFACTS,
        "save_prediction_artifacts": SAVE_PREDICTION_ARTIFACTS,
        "save_training_curves": SAVE_TRAINING_CURVES,
        "watch_model": WATCH_MODEL,
        "watch_log": WATCH_LOG,
        "watch_log_freq": WATCH_LOG_FREQ
    }
