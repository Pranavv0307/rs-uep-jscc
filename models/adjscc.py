"""
ADJSCC encoder/decoder (Xu et al. 2022, arXiv:2109.14467).

Same 5-conv / 5-transpose-conv backbone as models/deepjscc.py (kept
identical on purpose -- ADJSCC's contribution is the *attention* on top
of a plain JSCC backbone, not a different backbone), with one AFModule
inserted after every conv/transpose-conv layer, conditioned on the
current channel SNR.

Kept deliberately separate from deepjscc.py rather than editing it in
place:
  - Week 3's baseline (DeepJSCCEncoder/Decoder) stays as a frozen,
    already-verified reference to benchmark ADJSCC against later.
  - The RS-tiering work (Phase 4) reads attention weights specifically
    off this module's bottleneck AF gate -- keeping it a separate class
    means that hook is unambiguous.

NOTE: AF-module placement (post-conv/pre-activation here) and the
choice to condition *every* layer rather than only the bottleneck are
both design decisions -- re-check against Fig. 2 of the paper before
calling this a faithful reproduction; flag any discrepancy in the
Week 4 meeting note per the deliverables plan.
"""
import torch
import torch.nn as nn

from models.attention import AFModule
from models.deepjscc import power_normalize


class ADJSCCEncoder(nn.Module):
    def __init__(self, k_over_n: float = 1 / 6, image_size: int = 32):
        super().__init__()
        n = image_size * image_size * 3
        self.k = max(1, round(k_over_n * n))
        self._out_spatial = image_size // 4
        c_last = (2 * self.k) // (self._out_spatial * self._out_spatial)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.af1 = AFModule(16)
        self.act1 = nn.PReLU()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.af2 = AFModule(32)
        self.act2 = nn.PReLU()

        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.af3 = AFModule(32)
        self.act3 = nn.PReLU()

        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.af4 = AFModule(32)
        self.act4 = nn.PReLU()

        # bottleneck: no activation after this one, matching deepjscc.py.
        # This AF module's gate is the one Week 5 extracts as the
        # per-channel importance signal.
        self.conv5 = nn.Conv2d(32, c_last, kernel_size=5, stride=1, padding=2)
        self.af5 = AFModule(c_last)

    def forward(self, x: torch.Tensor, snr_db, return_attn: bool = False):
        attn = {} if return_attn else None

        x, g = self.af1(self.conv1(x), snr_db)
        if return_attn:
            attn["af1"] = g.detach()
        x = self.act1(x)

        x, g = self.af2(self.conv2(x), snr_db)
        if return_attn:
            attn["af2"] = g.detach()
        x = self.act2(x)

        x, g = self.af3(self.conv3(x), snr_db)
        if return_attn:
            attn["af3"] = g.detach()
        x = self.act3(x)

        x, g = self.af4(self.conv4(x), snr_db)
        if return_attn:
            attn["af4"] = g.detach()
        x = self.act4(x)

        x, g = self.af5(self.conv5(x), snr_db)
        if return_attn:
            attn["af5_bottleneck"] = g.detach()  # <- Week 5 / Phase 4 importance signal

        B = x.shape[0]
        z_tilde = x.view(B, -1)
        z = power_normalize(z_tilde, P=1.0)

        if return_attn:
            return z, attn
        return z


class ADJSCCDecoder(nn.Module):
    def __init__(self, k: int, image_size: int = 32):
        super().__init__()
        c_last = 2 * k
        spatial = image_size // 4
        self.k = k
        self.spatial = spatial
        c_in = c_last // (spatial * spatial)

        self.deconv1 = nn.ConvTranspose2d(c_in, 32, kernel_size=5, stride=1,
                                           padding=2, output_padding=0)
        self.af1 = AFModule(32)
        self.act1 = nn.PReLU()

        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.af2 = AFModule(32)
        self.act2 = nn.PReLU()

        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.af3 = AFModule(32)
        self.act3 = nn.PReLU()

        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2,
                                           padding=2, output_padding=1)
        self.af4 = AFModule(16)
        self.act4 = nn.PReLU()

        self.deconv5 = nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2,
                                           padding=2, output_padding=1)
        self.af5 = AFModule(3)
        self.out_act = nn.Sigmoid()

    def forward(self, z_hat_flat: torch.Tensor, snr_db, return_attn: bool = False):
        attn = {} if return_attn else None
        B = z_hat_flat.shape[0]
        c = z_hat_flat.shape[1] // (self.spatial * self.spatial)
        x = z_hat_flat.view(B, c, self.spatial, self.spatial)

        x, g = self.af1(self.deconv1(x), snr_db)
        if return_attn:
            attn["af1"] = g.detach()
        x = self.act1(x)

        x, g = self.af2(self.deconv2(x), snr_db)
        if return_attn:
            attn["af2"] = g.detach()
        x = self.act2(x)

        x, g = self.af3(self.deconv3(x), snr_db)
        if return_attn:
            attn["af3"] = g.detach()
        x = self.act3(x)

        x, g = self.af4(self.deconv4(x), snr_db)
        if return_attn:
            attn["af4"] = g.detach()
        x = self.act4(x)

        x, g = self.af5(self.deconv5(x), snr_db)
        if return_attn:
            attn["af5"] = g.detach()
        x_hat = self.out_act(x)

        if return_attn:
            return x_hat, attn
        return x_hat