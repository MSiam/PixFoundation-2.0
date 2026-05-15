from detectron2.config import CfgNode as CN
from detectron2.engine import default_setup
from detectron2.config import get_cfg


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """

    def merge_cfgs(current_dict, loaded_dict):
        for k, v in loaded_dict.items():
            if k in current_dict:
                if type(loaded_dict[k]) == dict or type(current_dict[k]) == CN:
                    current_dict[k] = merge_cfgs(current_dict[k], v)
                else:
                    current_dict[k] = v
            else:
                current_dict[k] = v
        return current_dict

    # FIXME: Detectron2 version allowing for new key instead of not found error in merge_from_config_file
    cfg = get_cfg()
    cfg_dict = dict(cfg)
    loaded_cfg_dict = cfg.load_yaml_with_base(args.config_file)
    cfg_dict = merge_cfgs(cfg_dict, loaded_cfg_dict)
    cfg = CN(cfg_dict)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


