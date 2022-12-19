from pathlib import Path
import yaml
import numpy as np
import torch
import hat.models
import torchvision.transforms as transforms
from basicsr.utils import imfrombytes, img2tensor, imwrite, tensor2img
from basicsr.utils.options import ordered_yaml
from basicsr.models import build_model

def hat_upscaler(path_opt_yaml):
    with open(path_opt_yaml, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    opt['is_train'] = False
    opt['dist'] = False
    opt['path']['pretrain_network_g'] = Path(__file__).parent.parent / opt['path']['pretrain_network_g']
    model = build_model(opt)
    transform = transforms.Compose([transforms.PILToTensor()])

    def upscale(img : np.ndarray, save_img_path, bgr2rgb=False):
        '''
        img = transform(img)
        img = img.to(torch.float32) / 255.
        img = torch.permute(img, (2,0,1))
        '''
        img = img2tensor(img, bgr2rgb=bgr2rgb, float32=True)
        img = img[None, :]
        model.feed_data({'lq':img})
        model.pre_process()
        if 'tile' in opt:
            model.tile_process()
        else:
            model.process()
        model.post_process()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, save_img_path)

    return upscale

if __name__ == '__main__':
    import sys
    config_file = 'HAT_SRx3-tiled.yml'
    config_file = Path(__file__).parent.parent / "options" / "run" / config_file
    upscale = hat_upscaler(config_file)
    if 1:
        from PIL import Image
        img = Image.open(sys.argv[-1])
        img = np.asarray(img).astype(np.float32) / 255.
        bgr2rgb = False
    else:
        img = imfrombytes(open(sys.argv[-1], "rb").read(), float32=True)
        bgr2rgb = True

    upscale(img, "out2.png", bgr2rgb=bgr2rgb)




