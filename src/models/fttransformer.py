from rtdl_revisiting_models import FTTransformer
from src.models.nn_train import NNWrapper


class FTModel(NNWrapper):
    def __init__(self, params, task, seed, meta, cfg):
        super().__init__(params, task, seed, meta, cfg, "fttransformer")

    def build(self):
        p = self.p
        kw = FTTransformer.get_default_kwargs(n_blocks=p["n_blocks"])
        kw.update(d_block=p["d_block"], attention_n_heads=4,
                  attention_dropout=p["attention_dropout"],
                  ffn_dropout=p["ffn_dropout"],
                  residual_dropout=p["residual_dropout"])
        self.net = FTTransformer(n_cont_features=self.meta["n_num"],
                                 cat_cardinalities=self.meta["cardinalities"],
                                 d_out=1, **kw)

    def param_groups(self):
        return self.net.make_parameter_groups()