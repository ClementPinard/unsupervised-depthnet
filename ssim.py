import torch
import torch.nn as nn
from torch.jit import ScriptModule, script_method, trace
import math

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def create_gaussian_window(window_size, channel):
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window@(_1D_window.t()).float()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIM(ScriptModule):
    def __init__(self, window_size=3):
        super(SSIM, self).__init__()

        gaussian_img_kernel = {'weight': create_gaussian_window(window_size, 3).float(),
                               'bias': torch.zeros(3)}
        gaussian_blur = nn.Conv2d(3,3,window_size, padding=window_size//2, groups=3).to(device)
        gaussian_blur.load_state_dict(gaussian_img_kernel)
        self.gaussian_blur = trace(gaussian_blur, torch.rand(3, 3, 16, 16, dtype=torch.float32, device=device))

    @script_method
    def forward(self, img1, img2):
        mu1 = self.gaussian_blur(img1)
        mu2 = self.gaussian_blur(img2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = self.gaussian_blur(img1*img1) - mu1_sq
        sigma2_sq = self.gaussian_blur(img2*img2) - mu2_sq
        sigma12 = self.gaussian_blur(img1*img2) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map