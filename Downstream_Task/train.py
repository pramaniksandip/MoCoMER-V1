from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
#from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.cli import LightningCLI

from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER

cli = LightningCLI(
    LitCoMER,
    CROHMEDatamodule,
    save_config_overwrite=True,
    trainer_defaults={"plugins": DDPPlugin(find_unused_parameters=False)},
    #trainer_defaults = {"strategy": DDPStrategy(find_unused_parameters=False)},
)
