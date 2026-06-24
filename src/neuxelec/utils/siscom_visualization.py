import matplotlib.cm as cm
import numpy as np


def get_siscom_window(values, zmin, zmax):
    if values is None:
        return float(zmin), float(max(zmin + 1e-6, zmax))

    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float(zmin), float(max(zmin + 1e-6, zmax))

    lo = float(zmin)
    hi = float(zmax)
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def normalize_siscom_slice(values, lo, hi, gamma=1.0, mask=None):
    values = np.asarray(values, dtype=np.float32)

    den = max(1e-6, float(hi) - float(lo))
    norm = (values - float(lo)) / den
    norm = np.clip(norm, 0.0, 1.0)

    gamma = max(0.1, float(gamma))
    if abs(gamma - 1.0) > 1e-6:
        norm = norm ** (1.0 / gamma)

    norm[~np.isfinite(values)] = 0.0

    if mask is not None:
        norm[np.asarray(mask) <= 0] = 0.0

    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    norm = np.clip(norm, 0.0, 1.0)
    return norm


def siscom_norm_to_colormap(siscom_norm, cmap_name="hot"):
    siscom_norm = np.asarray(siscom_norm, dtype=np.float32)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(np.clip(siscom_norm, 0.0, 1.0))[..., :3]
    return rgb.astype(np.float32)


def blend_siscom_on_rgba(base_rgba, siscom_rgb, siscom_alpha, alpha_scale=1.0):
    """
    Build an RGBA SISCOM overlay texture for display as a SEPARATE plane actor.
    RGB stores the colormap directly, and alpha controls transparency.
    """
    out = np.asarray(base_rgba, dtype=np.float32).copy()
    siscom_rgb = np.asarray(siscom_rgb, dtype=np.float32)

    if siscom_rgb.max() <= 1.0:
        siscom_rgb = siscom_rgb * 255.0

    alpha = np.clip(np.asarray(siscom_alpha, dtype=np.float32) * float(alpha_scale), 0.0, 1.0)
    mask = alpha > 0.0

    for c in range(3):
        out[..., c][mask] = siscom_rgb[..., c][mask]

    if out.shape[-1] >= 4:
        out[..., 3][mask] = 255.0 * alpha[mask]

    out = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)
