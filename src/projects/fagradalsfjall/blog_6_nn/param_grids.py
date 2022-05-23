from src.base.forecasting.models import Activation, LogLogAUCLoss, LrMaxCriterion, MSELoss

# -------------------------------------------------------------------------
#  Grid Search - Single Loss Function
# -------------------------------------------------------------------------
params_grid_search = {
    "lr_max": [LrMaxCriterion.VALLEY],
    "n_epochs": [10, 20, 50, 100, 200, 500],
    "wd": [0.0],
    "dropout": [0.0, 0.1, 0.2],
    "n_hidden_layers": [1, 2, 3],
    "layer_width": [100],
    "p": [64],
    "n": [1, 192],
}


# -------------------------------------------------------------------------
#  Informed Search - Multiple Loss Functions
# -------------------------------------------------------------------------
params_informed_search = {
    "activation": [Activation.ELU, Activation.RELU, Activation.SELU, Activation.GELU],
    "loss": [MSELoss(), LogLogAUCLoss()],
    "lr_max": [LrMaxCriterion.VALLEY, LrMaxCriterion.MINIMUM],
    "n_epochs": [5, 10, 20, 50, 100, 250, 500, 750, 1000, 1500],
    "batch_size": [64, 128, 256, 512, 1024],
    "wd": [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "n_hidden_layers": [1, 2, 3, 4, 5, 6],
    "layer_width": [25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500],
    "p": [8, 16, 32, 48, 64, 96, 128],
    "n": [1, 2, 4, 8, 16, 24, 48, 96, 144, 192],
}
