"""Vision policy: ResNet-18 (ImageNet init) on a 96x96 RGB frame + proprio (5) -> action chunk (K x 4).
Gripper is a separate sigmoid head (open/close is binary in the demos). Position deltas are regressed with L1.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
import numpy as np
from .env import PROPRIO_DIM, GOAL_DIM

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class BCPolicy(nn.Module):
    """use_goal=False: the policy must find BOTH the cube and the red goal marker in the image.
    use_goal=True: the goal xyz (the task command) is appended to proprio; the cube must still be found in the image."""
    def __init__(self, chunk: int = 1, pretrained: bool = True, use_goal: bool = False, hidden: int = 256):
        super().__init__()
        self.chunk, self.use_goal = chunk, use_goal
        proprio_dim = PROPRIO_DIM + (GOAL_DIM if use_goal else 0)
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.backbone = nn.Sequential(*list(r.children())[:-1])     # -> (B, 512, 1, 1)
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(512 + 64, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pos = nn.Linear(hidden, chunk * 3)
        self.grip = nn.Linear(hidden, chunk)
        self.register_buffer("p_mean", torch.zeros(proprio_dim)); self.register_buffer("p_std", torch.ones(proprio_dim))

    def forward(self, img_u8: torch.Tensor, proprio: torch.Tensor):
        """img_u8: (B, H, W, 3) uint8. proprio: (B, 5) float. Returns pos (B, K, 3) in [-1,1], grip logits (B, K)."""
        x = img_u8.permute(0, 3, 1, 2).float() / 255.0
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        f = self.backbone(x).flatten(1)
        p = self.proprio((proprio - self.p_mean) / self.p_std)
        h = self.head(torch.cat([f, p], 1))
        return torch.tanh(self.pos(h)).view(-1, self.chunk, 3), self.grip(h)

    @torch.no_grad()
    def act_chunk(self, img_u8: np.ndarray, proprio: np.ndarray, device) -> np.ndarray:
        pos, g = self(torch.from_numpy(np.ascontiguousarray(img_u8)[None]).to(device), torch.from_numpy(proprio[None]).to(device))
        grip = torch.where(torch.sigmoid(g) > 0.5, 1.0, -1.0)      # demos use +1 open / -1 close
        return torch.cat([pos[0], grip[0, :, None]], 1).cpu().numpy().astype(np.float32)


def loss_fn(pos_pred, grip_logit, action_chunk, mask):
    """action_chunk: (B, K, 4); mask: (B, K) 1 where the chunk step exists (episode not over)."""
    l_pos = (F.l1_loss(pos_pred, action_chunk[..., :3], reduction="none").mean(-1) * mask).sum() / mask.sum()
    tgt = (action_chunk[..., 3] > 0).float()
    l_grip = (F.binary_cross_entropy_with_logits(grip_logit, tgt, reduction="none") * mask).sum() / mask.sum()
    return l_pos + 0.5 * l_grip, l_pos.detach(), l_grip.detach()


class BCPolicyRunner:
    """Closed-loop wrapper used by eval.py: predicts a chunk, executes `exec_steps` of it, then re-plans."""
    def __init__(self, ckpt: str, device: str = "cuda", exec_steps: int | None = None):
        ck = torch.load(ckpt, map_location=device, weights_only=True)
        self.use_goal = ck.get("use_goal", False)
        self.model = BCPolicy(chunk=ck["chunk"], pretrained=False, use_goal=self.use_goal).to(device).eval()
        self.model.load_state_dict(ck["state_dict"]); self.device = device
        self.exec_steps = exec_steps or ck.get("exec_steps", ck["chunk"]); self.buf = []
        self.chunk = ck["chunk"]
        assert self.exec_steps <= self.chunk, f"cannot execute {self.exec_steps} steps from a {self.chunk}-step chunk"

    def reset(self): self.buf = []

    def act(self, obs, img):
        from .env import policy_vec
        if not self.buf:
            self.buf = list(self.model.act_chunk(img, policy_vec(obs, self.use_goal), self.device)[: self.exec_steps])
        return self.buf.pop(0)
