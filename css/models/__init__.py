# Copyright 2021
# Apache 2.0

import os
import glob
import importlib


modules = glob.glob(os.path.sep.join([os.path.dirname(__file__), "*.py"]))

for f in modules:
    if os.path.isfile(f) and "__init__.py" not in f:
        module_name, ext = os.path.splitext(f)
        if ext == ".py":
            module = importlib.import_module(
                "css.models." + os.path.basename(module_name)
            )

MODELS = {
    "Conformer": conformer.Conformer,
    "ConvTasNet": conv_tasnet.ConvTasNet,
}


def build_model(modelname, conf):
    return MODELS[modelname].build_model(conf)
