import os.path

from mmengine.config import Config
from mmengine.runner import Runner

dataset = "VTUS"

# path to results
root = ''

config_path = (f'./config/MWNet.py')
config = Config.fromfile(config_path)
root_path = (os.path.join(root,dataset,'MWNet'))
config.work_dir = root_path
runner = Runner.from_cfg(config)

if __name__ == '__main__':
    runner.train()