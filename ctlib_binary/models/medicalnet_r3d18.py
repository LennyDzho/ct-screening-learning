import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class R3D18Binary(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r3d_18(weights=weights)

        # адаптируем stem под 1 канал
        old_conv1 = self.backbone.stem[0]  # nn.Conv3d(3, 64, ...)
        new_conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        with torch.no_grad():
            if weights is not None:
                # усредняем по каналам RGB -> 1ch
                new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv1.weight, mode="fan_out", nonlinearity="relu")
        self.backbone.stem[0] = new_conv1

        # бинарная голова
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,D,H,W] -> logits [B]
        return self.backbone(x).squeeze(1)
