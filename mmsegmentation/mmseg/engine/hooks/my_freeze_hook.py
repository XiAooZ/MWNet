from mmengine.hooks import Hook
import re
from mmengine.registry import HOOKS
@HOOKS.register_module()
class CustomFreezeHook(Hook):
    def __init__(self, module_names=None, param_patterns=None):
        """
        Args:
            module_names (list[str]): 模块路径列表，比如 ['backbone', 'gated_memory.ffn1']
            param_patterns (list[str]): 支持正则表达式的参数名，比如 ['.*bias', '.*q_proj.*']
        """
        self.module_names = module_names or []
        self.param_patterns = param_patterns or []

    def before_train(self, runner):
        model = runner.model
        frozen_names = set()

        # 冻结指定模块
        for name in self.module_names:
            module = self._get_submodule(model, name)
            if module is not None:
                for n, p in module.named_parameters():
                    p.requires_grad = False
                    frozen_names.add(f"{name}.{n}")
            else:
                runner.logger.warning(f'[CustomFreezeHook] Module "{name}" not found.')

        # 冻结匹配的参数名
        for name, param in model.named_parameters():
            for pattern in self.param_patterns:
                if re.fullmatch(pattern, name):
                    param.requires_grad = False
                    frozen_names.add(name)

        runner.logger.info(f"[CustomFreezeHook] Frozen parameters ({len(frozen_names)}):")
        for n in sorted(frozen_names):
            runner.logger.info(f"  - {n}")

    def _get_submodule(self, model, name):
        parts = name.split('.')
        submodule = model
        for part in parts:
            if not hasattr(submodule, part):
                return None
            submodule = getattr(submodule, part)
        return submodule