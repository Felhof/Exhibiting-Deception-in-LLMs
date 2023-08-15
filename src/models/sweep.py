import wandb


def hyperparameter_sweep(sweep_config=None, project="hyperparameter-optimization"):
    # Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric
    # This is just an example config
    default_sweep_config = {
        "method": "random",  # grid, random
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "epochs": {"values": [2, 5, 10]},
            "batch_size": {"values": [256, 128, 64, 32]},
            "dropout": {"values": [0.3, 0.4, 0.5]},
            "conv_layer_size": {"values": [16, 32, 64]},
            "weight_decay": {"values": [0.0005, 0.005, 0.05]},
            "learning_rate": {"values": [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]},
            "optimizer": {"values": ["adam", "nadam", "sgd", "rmsprop"]},
            "activation": {"values": ["relu", "elu", "selu", "softmax"]},
        },
    }

    if not sweep_config:
        sweep_config = default_sweep_config

    # Initialize a new sweep
    # Arguments:
    #     – sweep_config: the sweep config dictionary defined above
    #     – entity: Set the username for the sweep
    #     – project: Set the project name for the sweep
    entity = "detecting-and-mitigating-deception"
    # project = "hyperparameter-optimization"
    sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)

    # Initialize a new sweep
    # Arguments:
    #     – sweep_id: the sweep_id to run - this was returned above by wandb.sweep()
    #     – function: function that defines your model architecture and trains it
    wandb.agent(sweep_id, train)


def train():
    """
    Example training loop for hyperparam sweep.
    """
    # Default values for hyper-parameters we're going to sweep over
    config_defaults = {
        "epochs": 5,
        "batch_size": 128,
        "weight_decay": 0.0005,
        "learning_rate": 1e-3,
        "activation": "relu",
        "optimizer": "nadam",
        "hidden_layer_size": 128,
        "conv_layer_size": 16,
        "dropout": 0.5,
        "momentum": 0.9,
        "seed": 42,
    }

    # Initialize a new wandb run
    wandb.init(config=config_defaults)

    # Config holds and saves hyperparameters and inputs
    config = wandb.config

    ### Run training here
