from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class UnfreezeHook(Hook):
    def __init__(self, module_names, unfreeze_epoch):
        self.module_names = module_names
        self.unfreeze_epoch = unfreeze_epoch
        self._unfrozen = False

    def before_train_iter(self, runner, batch_idx,data_batch=None):
        if not self._unfrozen and runner.iter >= self.unfreeze_epoch:
            model = runner.model
            for name in self.module_names:
                module = self._get_submodule(model, name)
                if module:
                    for n, p in module.named_parameters():
                        p.requires_grad = True
                        print(f'[UnfreezeHook] Unfroze: {name}.{n}')
            self._unfrozen = True

    def _get_submodule(self, model, name):
        parts = name.split('.')
        submodule = model
        for part in parts:
            if not hasattr(submodule, part):
                print(f'[UnfreezeHook] Cannot find module: {name}')
                return None
            submodule = getattr(submodule, part)
        return submodule