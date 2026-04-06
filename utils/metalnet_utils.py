# utils/metalnet_utils.py
import os
import torch


def _strip_module_prefix(state_dict: dict) -> dict:
    """DataParallel 저장된 'module.' prefix 제거."""
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _extract_state_dict(ckpt_obj):
    """
    체크포인트 포맷에 상관없이 state_dict를 추출.
    지원 포맷: raw dict, {"state_dict": ...}, {"student": ...}, {"model": ...}, {"net": ...}
    """
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "student", "model", "net", "module"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def load_metalnet(ckpt_path: str, device="cuda"):
    """
    pretrain/MetalicNet/net.py의 UNet(7ch in_channels)을 로드해서 eval 모드로 반환.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[MetalNet] checkpoint not found: {ckpt_path}")

    from pretrain.MetalicNet.net import UNet

    net = UNet(in_channels=7).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _strip_module_prefix(_extract_state_dict(ckpt))

    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        print(f"[MetalNet] missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[MetalNet] unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    net.eval()
    return net


@torch.no_grad()
def predict_metal_map(metalnet, render_pkg: dict):
    """
    render_pkg에서 7채널 입력을 구성하여 metal probability map (1,H,W)을 반환.

    2DGS 렌더 패키지 키 (sam2pass와 다름):
      - 'albedomap'    (3, H, W)
      - 'roughnessmap' (1, H, W)
      - 'rend_normal'  (3, H, W)  ← 2DGS surfel normal (sam2pass의 'normal')
    """
    if metalnet is None:
        return None

    albedo  = render_pkg.get("albedomap",    None)
    rough   = render_pkg.get("roughnessmap", None)
    # 2DGS uses 'rend_normal'; fall back to 'normal' for compatibility
    normal  = render_pkg.get("rend_normal",  render_pkg.get("normal", None))

    if albedo is None or rough is None or normal is None:
        return None

    if rough.dim() == 2:
        rough = rough.unsqueeze(0)  # (1, H, W)

    # Ensure unit normals and consistent device
    import torch.nn.functional as F
    device = albedo.device
    normal = F.normalize(normal.to(device), dim=0)
    rough  = rough.to(device)

    x = torch.cat([albedo, rough, normal], dim=0).unsqueeze(0)  # (1, 7, H, W)
    metal = metalnet(x).squeeze(0)  # (1, H, W), sigmoid output from UNet
    return metal.clamp(0.0, 1.0)


@torch.no_grad()
def metalprob_to_f0_rgb(render_pkg: dict,
                        metal_prob: torch.Tensor,
                        dielectric_f0: float = 0.04):
    """
    Disney 관례에 따른 F0 RGB 맵 계산.
      - non-metal: F0 ≈ 0.04 (RGB 동일)
      - metal:     F0_rgb ≈ basecolor (specular tint)

    Args:
        metal_prob: (1, H, W) in [0, 1]
        dielectric_f0: base reflectance for non-metals (default 0.04)

    Returns:
        f0_rgb: (3, H, W)
    """
    if metal_prob is None:
        return None

    albedo = render_pkg.get("albedomap", None)
    if albedo is None:
        return None

    if metal_prob.dim() == 2:
        metal_prob = metal_prob.unsqueeze(0)  # (1, H, W)

    m = metal_prob.clamp(0.0, 1.0)
    a = albedo.clamp(0.0, 1.0)
    f0_dielectric = torch.full_like(a, float(dielectric_f0))

    return ((1.0 - m) * f0_dielectric + m * a).clamp(0.0, 1.0)
