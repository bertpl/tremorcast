from .tabular_regressor_mlp import TabularRegressorMLP
from .tabular_regressor_multi import TabularRegressorMulti


# =================================================================================================
#  "Multi-MLP" Tabular Model - Composed of multiple single-output MLP
# =================================================================================================
class TabularRegressorMultiMLP(TabularRegressorMulti):

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, n_inputs: int, n_outputs: int, **kwargs):
        regressors = [TabularRegressorMLP(n_inputs=n_inputs, n_outputs=1) for _ in range(n_outputs)]
        super().__init__("mlp-multi", regressors)
