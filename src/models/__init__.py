from src.models.linear import LinearModel
from src.models.rf import RFModel
from src.models.gbdt import XGBModel
from src.models.mlp import MLPModel
from src.models.fttransformer import FTModel

# name -> (class, preprocessing kind)
REGISTRY = {
    "linear": (LinearModel, "linear"),
    "rf": (RFModel, "tree"),
    "xgb": (XGBModel, "tree"),
    "mlp": (MLPModel, "neural"),
    "fttransformer": (FTModel, "neural"),
}


def make_model(name, params, task, seed, meta, cfg):
    return REGISTRY[name][0](params, task, seed, meta, cfg)