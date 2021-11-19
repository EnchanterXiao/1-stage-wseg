from .BaselineCAM import *
from .SoftMaxAE import *
from .CAM_SA import *
from .CAM_CASA import *
from .CAM_SA_WGAP import *
from .CAM_CASA_WGAP import *
from .CAM_MF import *
from .CAM_MF_v2 import *
from .CAM_CASA_WGAP_v2 import *
from .CAM_CASA_WGAP_v3 import *
from .CAM_WGAP_v3 import *
from .CAM_CASA_WGAP_v4 import *
from .CAM_CASA_WGAP_v5 import *
from .CAM_CASA_WGAP_PCM import *
from .CAM_CASA_WGAP_v6 import *
from .CAM_CASA_WGAP_tf import *
from .CAM_CASA_WGAP_tf_v2 import *

#
# Dynamic change of the base class
#
def network_factory(cfg):
    if cfg.MODEL == 'ae':
        print("Model: AE")
        return network_SoftMaxAE(cfg)
    elif cfg.MODEL == 'bsl':
        print("Model: Baseline")
        return network_BaselineCAM(cfg)
    elif cfg.MODEL == 'CAM_SA':
        return network_CAM_SA(cfg)
    elif cfg.MODEL == 'CAM_CASA':
        return network_CAM_CASA(cfg)
    elif cfg.MODEL == 'CAM_SA_WGAP':
        return network_CAM_SA_WGAP(cfg)
    elif cfg.MODEL == 'CAM_CASA_WGAP':
        return network_CAM_CASA_WGAP(cfg)
    elif cfg.MODEL == 'CAM_MF':
        return network_CAM_MF(cfg)
    elif cfg.MODEL == 'CAM_MF_v2':
        return network_CAM_MF_v2(cfg)
    elif cfg.MODEL == 'CAM_CASA_WGAP_v2':
        return network_CAM_CASA_WGAP_v2(cfg)
    elif cfg.MODEL == 'CAM_CASA_WGAP_v3':
        return network_CAM_CASA_WGAP_v3(cfg)
    elif cfg.MODEL == 'CAM_WGAP_v3':
        return network_CAM_WGAP_v3(cfg)
    elif cfg.MODEL == 'CAM_CASA_WGAP_v4':
        return network_CAM_CASA_WGAP_v4(cfg)
    elif cfg.MODEL == 'CAM_CASA_WGAP_v5':
        return network_CAM_CASA_WGAP_v5(cfg)
    elif cfg.MODEL == 'CAM_CASA_WGAP_PCM':
        return network_CAM_CASA_WGAP_PCM(cfg)
    elif cfg.MODEL == 'CAM_CASA_WGAP_v6':
        return network_CAM_CASA_WGAP_v6(cfg)
    elif cfg.MODEL == 'CAM_CASA_WGAP_tf':
        return network_CAM_CASA_WGAP_tf(cfg)
    elif cfg.MODEL == 'CAM_CASA_WGAP_tf_v2':
        return network_CAM_CASA_WGAP_tf_v2(cfg)
    else:
        raise NotImplementedError("Unknown model '{}'".format(cfg.MODEL))

if __name__ == '__main__':
    from core.config import cfg, cfg_from_file, cfg_from_list
    from functools import partial

    # Reading the config
    cfg_from_file('../configs/voc_resnet38.yaml')
    net = network_factory(cfg.NET)(cfg.NET)
    print("Config: \n", cfg)
    print(net)