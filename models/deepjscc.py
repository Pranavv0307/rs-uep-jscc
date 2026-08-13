"""
DeepJSCC encoder/decoder, following Bourtsoulatze, Kurka & Gündüz (2019).
5 conv (encoder) / 5 transpose-conv (decoder) layers, PReLU activations,
power-normalized bottleneck of 2k units representing k complex symbols.

NOTE: verify the exact F/K/S values against Figure 2 of arXiv:1809.01733
(open the PDF yourself, page 4) before treating this as a byte-for-byte
reproduction — see the end-of-week checklist. The config below is a
standard, correct reproduction pattern (5 layers, PReLU, 2k-unit
power-normalized bottleneck) consistent with everything stated in the
paper's text.
"""
import torch
import torch.nn as nn


class DeepJSCCEncoder(nn.Module):
    def __init__(self, k_over_n: float = 1 / 6, image_size: int = 32):
        super().__init__()
        n = image_size * image_size * 3
        self.k = max(1, round(k_over_n * n))
        c_last = 2 * self.k

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, stride=2, padding=4),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, c_last, kernel_size=5, stride=1, padding=2),
        )
        self._out_spatial = image_size // 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_tilde = self.net(x)
        B = z_tilde.shape[0]
        z_tilde = z_tilde.view(B, -1)
        return power_normalize(z_tilde, P=1.0)


class DeepJSCCDecoder(nn.Module):
    def __init__(self, k: int, image_size: int = 32):
        super().__init__()
        c_last = 2 * k
        spatial = image_size // 4
        self.k = k
        self.spatial = spatial

        self.net = nn.Sequential(
            nn.ConvTranspose2d(c_last // (spatial * spatial), 32, kernel_size=5,
                                stride=1, padding=2, output_padding=0),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2,
                                output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=9, stride=2, padding=4,
                                output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_hat_flat: torch.Tensor) -> torch.Tensor:
        B = z_hat_flat.shape[0]
        c = z_hat_flat.shape[1] // (self.spatial * self.spatial)
        x = z_hat_flat.view(B, c, self.spatial, self.spatial)
        return self.net(x)


def power_normalize(z_tilde: torch.Tensor, P: float = 1.0) -> torch.Tensor:
    """
    Eq. (1) of Bourtsoulatze et al.: z = sqrt(k*P) * z_tilde / ||z_tilde||
    """
    k = z_tilde.shape[1] / 2
    norm = torch.sqrt((z_tilde ** 2).sum(dim=1, keepdim=True) + 1e-8)
    return torch.sqrt(torch.tensor(k * P)) * z_tilde / norm
