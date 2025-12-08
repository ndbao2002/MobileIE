import torch
from importlib import import_module
from .lle import MobileIELLENet, MobileIELLENetS
from .isp import MobileIEISPNet, MobileIEISPNetS
from .isp_v2 import MobileRetinexISPNet, MobileRetinexISPNetS

__all__ = {
    'MobileIELLENet',
    'MobileIELLENetS',
    'MobileIEISPNet', 
    'MobileIEISPNetS',
    'MobileRetinexISPNet',
    'MobileRetinexISPNetS',
    'import_model'
}

def import_model(opt):
    model_name = 'MobileIE'+opt.model_task.upper()
    if opt.config['model']['name'] == 'retinex':
        model_name = 'MobileRetinex'+opt.model_task.upper()
    kwargs = {'channels': opt.config['model']['channels']}

    if opt.config['model']['type'] == 're-parameterized':
        model_name += 'NetS'
    elif opt.config['model']['type'] == 'original':
        model_name += 'Net'
        kwargs['rep_scale'] = opt.config['model']['rep_scale']
    else:
        raise ValueError('unknown model type, please choose from [original, re-parameterized]')

    model = getattr(import_module('model'), model_name)(**kwargs)
    model = model.to(opt.device)

    if opt.config['model']['pretrained']:
        if opt.task == 'train' and opt.config['model']['pretrained_date'] is not None:
            pretrained_path = r'{}/model_pre.pkl'.format(opt.save_model_dir)
            model.load_state_dict(torch.load(pretrained_path), strict=False)
        else:
            #model.load_state_dict(torch.load(opt.config['model']['pretrained']))
            model.load_state_dict(torch.load(opt.config['model']['pretrained']), strict=False)

    if opt.config['model']['type'] == 'original' and opt.config['model']['need_slim'] is True:
        model = model.slim().to(opt.device)
    return model
