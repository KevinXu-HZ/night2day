import torch, torch.nn as nn, torch.nn.functional as F, torchvision

class GANLoss(nn.Module):
    def __init__(self): super().__init__(); self.mse = nn.MSELoss()
    def forward(self, pred, is_real: bool):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.mse(pred, target)

class SemanticConsistencyLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.seg = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT").to(device).eval()
        for p in self.seg.parameters(): p.requires_grad = False
        self.mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        self.std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)

    @torch.no_grad()
    def _logits(self, x):
        x01 = (x.clamp(-1,1)+1)/2
        xnorm = (x01 - self.mean)/self.std
        return self.seg(xnorm)["out"]

    def forward(self, src, gen):
        with torch.no_grad():
            p_src = torch.softmax(self._logits(src), dim=1)
        p_gen = torch.softmax(self._logits(gen), dim=1)
        return F.l1_loss(p_gen, p_src)
