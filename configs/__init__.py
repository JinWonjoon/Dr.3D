import pdb
from configtree import Loader, Walker, Updater
import math

def walk_configs(config_path):
    walk = None
    update = Updater(namespace={'None': None, 'len': len, 'pi': math.pi})
    load = Loader(walk=walk, update=update)
    print(f'walking {config_path}...')
    main_configs = load(config_path)
    main_configs = main_configs.rare_copy() ## to dictionary
    return main_configs