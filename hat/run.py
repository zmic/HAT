from pathlib import Path
import yaml
import numpy as np
from basicsr.utils import imfrombytes, img2tensor, imwrite, tensor2img
from basicsr.utils.options import ordered_yaml
from basicsr.models import build_model
import hat.models

def hat_upscaler(path_opt_yaml, save_function=False):
    hat_root = Path(__file__).parent.parent
    path_opt_yaml = hat_root / "options" / "run" / path_opt_yaml

    with open(path_opt_yaml, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    opt['is_train'] = False
    opt['dist'] = False
    # cwd may be anything, so rebase the path to the model file
    opt['path']['pretrain_network_g'] = hat_root / opt['path']['pretrain_network_g']
    model = build_model(opt)

    def upscale(img : np.ndarray, bgr2rgb=False):
        img = img2tensor(img, bgr2rgb=bgr2rgb, float32=True)
        img = img[None, :]
        model.feed_data({'lq':img})
        model.pre_process()
        if 'tile' in opt:
            model.tile_process()
        else:
            model.process()
        model.post_process()

        sr_img = model.get_current_visuals()['result']
        sr_img = tensor2img([sr_img])
        return sr_img

    if save_function:
        def upscale_save(img : np.ndarray, save_img_path, bgr2rgb=False):
            sr_img = upscale(img, save_img_path, bgr2rgb)
            imwrite(sr_img, str(save_img_path))
        return upscale_save

    return upscale

if __name__ == '__main__':
    import sys
    config_file = 'HAT_SRx3-tiled.yml'
    upscale_save = hat_upscaler(config_file, save_function=True)
    if 1:
        # use this method if you already have an PIL or numpy image in memory
        from PIL import Image
        img = Image.open(sys.argv[-1])
        img = np.asarray(img).astype(np.float32) / 255.
        bgr2rgb = False
    else:
        img = imfrombytes(open(sys.argv[-1], "rb").read(), float32=True)
        bgr2rgb = True

    upscale_save(img, "out2.jpg", bgr2rgb=bgr2rgb)




