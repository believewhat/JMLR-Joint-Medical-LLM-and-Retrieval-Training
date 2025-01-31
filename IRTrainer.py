import os
import torch
import torch.nn as nn

from transformers import Trainer
from typing import Dict, Optional, Sequence
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import safetensors.torch

from transformers.modeling_utils import PreTrainedModel, unwrap_model

from peft import PeftModel
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_peft_available,
    is_sagemaker_mp_enabled,
    logging,
)
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from colbert.parameters import SAVED_CHECKPOINTS
from colbert.infra.run import Run
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
TRAINING_ARGS_NAME = "training_args.bin"
logger = logging.get_logger(__name__)





class LlamaIRTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        self.model.model.model.colbert.save(os.path.join(output_dir, "ir"))
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    
    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_param_names = set(self.get_decay_parameter_names(opt_model))
            decay_parameters = self.get_decay_parameter_names(opt_model)


            param_groups = [
                {'params': [], 'lr': self.args.learning_rate, 'weight_decay': self.args.weight_decay},
                {'params': [], 'lr': self.args.learning_rate, 'weight_decay': 0.0},
                {'params': [], 'lr': self.args.lr_colbert, 'weight_decay': self.args.weight_decay},
                {'params': [], 'lr': self.args.lr_colbert, 'weight_decay': 0.0},
            ]

            # Assign parameters to the appropriate group
            for name, param in opt_model.named_parameters():
                if param.requires_grad:
                    if name.startswith('base_model.model.colbert.'):
                        if name in decay_param_names:
                            param_groups[0]['params'].append(param)
                        else:
                            param_groups[1]['params'].append(param)
                    else:
                        if name in decay_param_names:
                            param_groups[2]['params'].append(param)
                        else:
                            param_groups[3]['params'].append(param)

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer