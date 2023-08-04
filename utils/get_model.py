# Models
from pytorch_models.dgcnn import DGCNN
from pytorch_models.regressor import Regressor
from pytorch_models.ocnn_hrnet import HRNet

# Function that returns the specified model in the config (cfg) file
def get_model(cfg):

    if cfg['model_name'] == 'DGCNN':
        model = DGCNN(dropout=cfg['dropout_probability'], k=cfg['DGCNN_k'])

    elif cfg['model_name'] == 'OCNN_HRNet':
            #Set the num_stages
            if cfg['ocnn_stages'] == "auto":
                cfg['ocnn_stages'] = cfg['octree_depth'] - 2

            # Set number of lidar attributes in input and include normals in input if need be
            if cfg['ocnn_use_additional_features'] and cfg['use_normals']:
                in_channels = 9
            elif cfg['ocnn_use_additional_features']:
                in_channels = 8
            else:
                in_channels = 3

            model = HRNet(in_channels=in_channels, out_channels=1024,
                          stages=cfg['ocnn_stages'], interp='linear', nempty=True, 
                          dropout=cfg['dropout_probability'])
    
    else:
        raise Exception("No model architecture selected")

    # WRAP MODEL IN REGRESSOR SO EACH ARCHITECTURE HAS SAME OUTPUT MLP LAYER

    if cfg['target'] == "biomass_comps":
        n_outs = 4
    elif cfg['target'] == "total_agb":
        n_outs = 1
    else:
        raise Exception(f"Target: {cfg['target']} is not supported")

    model = Regressor(model=model, num_outputs=n_outs)

    return model
