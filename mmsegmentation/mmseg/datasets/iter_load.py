from mmengine.registry import  LOOPS
from mmengine.runner.loops import IterBasedTrainLoop
import logging
from mmengine.logging import HistoryBuffer, print_log
import mmengine.runner.loops
@LOOPS.register_module()
class RefreshIterBasedTransLoop(IterBasedTrainLoop):
    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        if self._iter > 0:
            print_log(
                f'Advance dataloader {self._iter} steps to skip data '
                'that has already been trained',
                logger='current',
                level=logging.WARNING)
            for _ in range(self._iter):
                next(self.dataloader_iterator)
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            if(self._iter % 2000 == 0 and self._iter != 0):
                print_log(
                    f'Advance dataloader {self._iter} steps to refresh data '
                    'that has already been trained',
                    logger='current',
                    level=logging.WARNING)
                self.dataloader = self.runner.build_dataloader(self.runner.cfg.train_dataloader)
                self.dataloader_iterator = iter(self.dataloader)

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model