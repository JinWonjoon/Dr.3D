import wandb
import pdb

class wandblogger(object):
    def __init__(self, cfg):
        wandb_entity = cfg.wandb_cfg['wandb_entity']
        cfg.wandb_cfg['wandb_id'] = wandb.util.generate_id() if cfg.wandb_cfg['wandb_id'] is None else cfg.wandb_cfg['wandb_id']
        wandb.init(project=cfg.wandb_cfg['wandb_project'], entity=wandb_entity, name=cfg.wandb_cfg['wandb_name'], id=cfg.wandb_cfg['wandb_id'], resume="allow", dir=cfg.run_dir)
        wandb.config.update(cfg, allow_val_change=True)

    def finish(self):
        wandb.finish()

    def log_dicts(self, dicts, step):
        wandb.log(dicts, step=step)
    
    def log_images(self, log_dict, step):
        print("log image to wandb")
        image_log = {}
        for name, dicts in log_dict.items():
            image_log[name] = wandb.Image(dicts['image'], caption=dicts['caption'])
        wandb.log(image_log, step=step)
        
    