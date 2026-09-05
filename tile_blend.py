"""Core tile-blending engine.

Given two terrain tile images (a "base" terrain A and an "overlay" terrain B),
generates the set of transition tiles needed to place B tiles next to A tiles
on a tile grid: one tile per corner combination (which corners belong to A vs
B), plus two special tiles for a lone A tile fully surrounded by B ("island")
and a lone B tile fully surrounded by A ("pond").

No GUI dependencies here -- see tile_generator_gui.py for the Tkinter front end.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

RESAMPLE_METHODS = {
    "Nearest": Image.NEAREST,
    "Bilinear": Image.BILINEAR,
    "Lanczos": Image.LANCZOS,
}

# Each combo: (code, (tl, tr, bl, br), category)  -- 0 = terrain A, 1 = terrain B
STANDARD_COMBOS = [
    ("corner-br", (0, 0, 0, 1), "corner"),
    ("edge-bottom", (0, 0, 1, 1), "edge"),
    ("corner-bl", (0, 0, 1, 0), "corner"),
    ("edge-right", (0, 1, 0, 1), "edge"),
    ("edge-left", (1, 0, 1, 0), "edge"),
    ("corner-tr", (0, 1, 0, 0), "corner"),
    ("edge-top", (1, 1, 0, 0), "edge"),
    ("corner-tl", (1, 0, 0, 0), "corner"),
    ("inner-tl", (0, 1, 1, 1), "inner"),
    ("inner-tr", (1, 0, 1, 1), "inner"),
    ("inner-bl", (1, 1, 0, 1), "inner"),
    ("inner-br", (1, 1, 1, 0), "inner"),
]

DIAGONAL_COMBOS = [
    ("diag-tlbr", (1, 0, 0, 1), "diagonal"),
    ("diag-trbl", (0, 1, 1, 0), "diagonal"),
]

# Exact slot order reproducing the reference 037-051 water tileset when
# start_index=37 and the overlay stem is "water" (see sequential_filenames).
LEGACY_ORDER = [
    "corner-br", "edge-bottom", "corner-bl", "edge-right",
    "__PURE_B__",
    "edge-left", "corner-tr", "edge-top", "corner-tl",
    "island",
    "inner-tl", "inner-tr", "inner-bl", "inner-br",
    "pond",
]


@dataclass
class BlendOptions:
    size: int = 48
    resample: str = "Lanczos"
    noise_strength: float = 0.35
    feather_px: float = 2.0
    edge_margin_frac: float = 0.18
    include_diagonals: bool = False
    include_specials: bool = True
    blob_radius_frac: float = 0.28
    blob_feather_frac: float = 0.10
    seed: int = 0
    transition_width_px: float = 0.0  # 0 disables the transition band


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")


def resize_to(img: Image.Image, size: int, resample: str) -> Image.Image:
    method = RESAMPLE_METHODS.get(resample, Image.LANCZOS)
    if img.size == (size, size):
        return img.copy()
    return img.resize((size, size), method)


def derive_seed(seed: int, code: str) -> int:
    return (int(seed) * 1_000_003 + zlib.crc32(code.encode("utf-8"))) & 0xFFFFFFFF


def _value_noise(size: int, seed: int, blur_radius: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.random((size, size)).astype(np.float32)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=max(0.5, blur_radius)))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = a - a.mean()
    max_abs = float(np.max(np.abs(a))) + 1e-6
    return a / max_abs


def _edge_fade_window(size: int, margin_frac: float) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.minimum(np.minimum(xs, size - 1 - xs), np.minimum(ys, size - 1 - ys))
    margin = max(1.0, size * margin_frac)
    w = np.clip(d / margin, 0.0, 1.0)
    return w * w * (3 - 2 * w)


def _corner_gradient(size: int, tl: float, tr: float, bl: float, br: float) -> np.ndarray:
    u = np.linspace(0.0, 1.0, size, dtype=np.float32)
    v = np.linspace(0.0, 1.0, size, dtype=np.float32)
    U, V = np.meshgrid(u, v)
    top = tl * (1 - U) + tr * U
    bottom = bl * (1 - U) + br * U
    return top * (1 - V) + bottom * V


def _smoothstep_mask(values: np.ndarray, center: float, width: float) -> np.ndarray:
    width = max(width, 1e-3)
    lo, hi = center - width / 2, center + width / 2
    t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _falloff(x: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """1.0 for x <= inner, 0.0 for x >= outer, smooth in between."""
    outer = max(outer, inner + 1e-6)
    t = np.clip((x - inner) / (outer - inner), 0.0, 1.0)
    return 1.0 - t * t * (3 - 2 * t)


def _keep_largest_components(binary: np.ndarray, count: int = 1) -> np.ndarray:
    """Removes every connected component of `binary` except the `count` largest."""
    labeled, n = ndimage.label(binary)
    if n <= count:
        return binary
    sizes = ndimage.sum(binary, labeled, index=range(1, n + 1))
    keep = np.argsort(sizes)[::-1][:count] + 1
    return np.isin(labeled, keep)


def _clean_disconnected_specks(binary: np.ndarray, expected_components: int = 1) -> np.ndarray:
    """Collapses a noisy binary split into `expected_components` region(s) per side.

    The underlying gradient is a single smooth saddle (or, for the diagonal
    combos, a saddle with two opposite-corner regions per side), so there
    should only ever be that many coherent regions per side; anything beyond
    that is a noise artifact (a fleck of B floating inside A, or vice versa)
    and gets reassigned to its surrounding majority.
    """
    binary = _keep_largest_components(binary, expected_components)          # drop stray B flecks
    binary = ~_keep_largest_components(~binary, expected_components)        # drop stray A flecks (holes in B)
    return binary


def build_corner_mask(size: int, tl: int, tr: int, bl: int, br: int,
                       noise_strength: float, edge_margin_frac: float,
                       feather_width: float, seed: int,
                       expected_components: int = 1) -> np.ndarray:
    g = _corner_gradient(size, tl, tr, bl, br)
    n = _value_noise(size, seed, blur_radius=max(1.0, size * 0.12))
    edge_w = _edge_fade_window(size, edge_margin_frac)

    # Gate the noise by how close each pixel already is to the 0.5 midline.
    # Without this, the noise field (normalized so it always reaches its
    # +/-1 extreme somewhere in the tile) could flip pixels that are far
    # from the intended coastline -- e.g. deep in solid "corner" or "inner"
    # combos where the baseline gradient sits well away from 0.5 -- creating
    # disconnected speckle/fleck artifacts unrelated to the actual boundary.
    # Capping the band at a fixed fraction (independent of noise_strength)
    # keeps most flips genuinely near the midline.
    band_outer = min(max(noise_strength, 0.05), 0.25)
    band_inner = band_outer * 0.55
    proximity_w = _falloff(np.abs(g - 0.5), band_inner, band_outer)

    total_w = edge_w * proximity_w
    perturbed = np.clip(g + noise_strength * n * total_w, 0.0, 1.0)

    # The actual antialiased edge comes straight from the continuous field,
    # so it can be feathered arbitrarily finely (including sub-pixel widths)
    # -- rebuilding it from a thresholded binary via a distance transform
    # would quantize it to whole pixels and make any feather <= ~2px a no-op.
    soft_mask = _smoothstep_mask(perturbed, 0.5, feather_width)

    # Belt-and-braces: remove any remaining disconnected fleck by keeping
    # only the single largest region on each side, then hard-patch just the
    # (typically handful of) pixels that reassignment actually touched --
    # everywhere else keeps its finely feathered value from above.
    raw_binary = perturbed >= 0.5
    cleaned_binary = _clean_disconnected_specks(raw_binary, expected_components)
    flipped = cleaned_binary != raw_binary
    if np.any(flipped):
        soft_mask = np.where(flipped, cleaned_binary.astype(soft_mask.dtype), soft_mask)
    return soft_mask


def build_blob_mask(size: int, invert: bool, radius_frac: float, feather_frac: float,
                     noise_strength: float, seed: int) -> np.ndarray:
    cx = cy = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    dist = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    rng = np.random.default_rng(seed)
    harmonics = rng.uniform(-1.0, 1.0, size=6)
    ang_noise = np.zeros_like(theta)
    for k in range(1, 4):
        ang_noise += harmonics[2 * (k - 1)] * np.cos(k * theta) + harmonics[2 * (k - 1) + 1] * np.sin(k * theta)
    ang_noise /= 3.0

    half = size * 0.5
    max_extent = half * 0.92
    radius = np.clip(radius_frac * half, 2.0, max_extent * 0.85)
    feather = np.clip(feather_frac * half, 1.0, max_extent - radius)
    radius_field = radius * (1.0 + 0.5 * noise_strength * ang_noise)
    radius_field = np.clip(radius_field, 1.0, max_extent - feather)

    t = np.clip((dist - (radius_field - feather)) / (2 * feather), 0.0, 1.0)
    blob = 1.0 - (t * t * (3 - 2 * t))  # 1 inside, 0 outside
    return (1.0 - blob) if invert else blob


def composite(img_a: Image.Image, img_b: Image.Image, mask: np.ndarray) -> Image.Image:
    a = np.asarray(img_a, dtype=np.float32)
    b = np.asarray(img_b, dtype=np.float32)
    m = mask[..., None]
    out = a * (1 - m) + b * m
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGBA")


def _signed_distance(binary: np.ndarray) -> np.ndarray:
    """Euclidean distance to the A/B boundary: positive inside B, negative inside A."""
    dist_into_b = distance_transform_edt(binary)
    dist_into_a = distance_transform_edt(~binary)
    return dist_into_b - dist_into_a


def _band_weights(sd: np.ndarray, half_width: float, feather: float = 0.75):
    lo_edge, hi_edge = -half_width, half_width
    a_weight = _falloff(sd, lo_edge - feather, lo_edge + feather)
    b_weight = 1.0 - _falloff(sd, hi_edge - feather, hi_edge + feather)
    a_weight = np.clip(a_weight, 0.0, 1.0)
    b_weight = np.clip(b_weight, 0.0, 1.0)
    c_weight = np.clip(1.0 - a_weight - b_weight, 0.0, 1.0)
    total = a_weight + b_weight + c_weight
    total = np.where(total < 1e-6, 1.0, total)
    return a_weight / total, c_weight / total, b_weight / total


def composite_with_transition(img_a: Image.Image, img_b: Image.Image, img_c: Image.Image,
                               mask: np.ndarray, band_px: float, feather_px: float = 1.5) -> Image.Image:
    """Like composite(), but inserts a slim band of img_c along the A/B boundary."""
    binary = mask >= 0.5
    sd = _signed_distance(binary)
    aw, cw, bw = _band_weights(sd, max(band_px, 0.5) / 2.0, feather=max(feather_px, 0.5) / 2.0)
    a = np.asarray(img_a, dtype=np.float32)
    b = np.asarray(img_b, dtype=np.float32)
    c = np.asarray(img_c, dtype=np.float32)
    out = a * aw[..., None] + c * cw[..., None] + b * bw[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGBA")


def generate_full_set(img_a: Image.Image, img_b: Image.Image, opts: BlendOptions,
                       img_c: "Image.Image | None" = None) -> dict:
    """Returns dict[code -> PIL.Image] including '_pure_a' and '_pure_b'.

    If img_c (a transition texture, e.g. sand) is given and opts.transition_width_px > 0,
    a slim band of img_c is inserted along every A/B boundary instead of a direct blend.
    """
    size = opts.size
    a = resize_to(img_a, size, opts.resample)
    b = resize_to(img_b, size, opts.resample)
    c = resize_to(img_c, size, opts.resample) if (img_c is not None and opts.transition_width_px > 0) else None

    def blend(mask):
        if c is not None:
            return composite_with_transition(a, b, c, mask, opts.transition_width_px, opts.feather_px)
        return composite(a, b, mask)

    feather_width = max(opts.feather_px, 0.5) / size

    combos = list(STANDARD_COMBOS)
    if opts.include_diagonals:
        combos = combos + DIAGONAL_COMBOS

    results = {}
    for code, bits, _cat in combos:
        tl, tr, bl, br = bits
        seed = derive_seed(opts.seed, code)
        expected_components = 2 if _cat == "diagonal" else 1
        mask = build_corner_mask(size, tl, tr, bl, br, opts.noise_strength,
                                  opts.edge_margin_frac, feather_width, seed,
                                  expected_components)
        results[code] = blend(mask)

    if opts.include_specials:
        seed_island = derive_seed(opts.seed, "island")
        mask_island = build_blob_mask(size, True, opts.blob_radius_frac,
                                       opts.blob_feather_frac, opts.noise_strength, seed_island)
        results["island"] = blend(mask_island)

        seed_pond = derive_seed(opts.seed, "pond")
        mask_pond = build_blob_mask(size, False, opts.blob_radius_frac,
                                     opts.blob_feather_frac, opts.noise_strength, seed_pond)
        results["pond"] = blend(mask_pond)

    results["_pure_a"] = a
    results["_pure_b"] = b
    if c is not None:
        results["_pure_c"] = c
    return results


def to_isometric(img: Image.Image) -> Image.Image:
    """Converts a square top-down tile into a diamond-shaped isometric tile.

    Rotates the square -45 degrees then squashes it vertically by half,
    producing the classic 2:1 isometric diamond silhouette with
    transparent padding filling the corners of the canvas.

    The rotation direction matters: it must send the square's top/right/
    bottom/left edges to the diamond's NE/SE/SW/NW edges respectively, so
    each edge ends up touching the same neighbor it touched on the square
    grid (see _build_isometric_sheet's (c, r) -> screen position formula,
    where +c moves SE and +r moves SW). Rotating +45 instead sends them to
    NW/NE/SE/SW -- one position off -- which misaligns every transition
    tile's boundary with its isometric neighbor, breaking the fit.
    """
    img = img.convert("RGBA")
    rotated = img.rotate(-45, expand=True, resample=Image.BICUBIC)
    w, h = rotated.size
    return rotated.resize((w, max(1, round(h / 2))), Image.LANCZOS)


EXAMPLE_LAYOUT_ROWS = 10
EXAMPLE_LAYOUT_COLS = 12

_BITS_TO_CODE = {bits: code for code, bits, _cat in STANDARD_COMBOS + DIAGONAL_COMBOS}


def example_corner_grid(rows: int, cols: int, include_diagonals: bool) -> list:
    """Builds a static 0/1 corner-ownership grid ((rows+1) x (cols+1) corners).

    Every 2x2 window of the grid is one tile's (tl, tr, bl, br) corner combo
    (0 = terrain A, 1 = terrain B). The shape below is hand-tuned so every
    STANDARD_COMBOS code appears in at least one window, and both
    DIAGONAL_COMBOS codes appear too when `include_diagonals` is set.
    """
    r, c = rows + 1, cols + 1
    grid = [[0] * c for _ in range(r)]

    def fill(r0, r1, c0, c1, value):
        for y in range(r0, r1 + 1):
            for x in range(c0, c1 + 1):
                grid[y][x] = value

    # A big square with a notch bitten out of each corner: gives every
    # convex "corner" and straight "edge" combo around its outside, and
    # every concave "inner" combo where a notch meets the body.
    fill(2, 8, 3, 9, 1)
    fill(2, 3, 3, 4, 0)
    fill(2, 3, 8, 9, 0)
    fill(7, 8, 3, 4, 0)
    fill(7, 8, 8, 9, 0)

    if include_diagonals:
        # Two isolated 2x2 checkerboard patches: the only shape that
        # produces the "opposite corners" diagonal combos.
        grid[r - 3][1], grid[r - 3][2] = 1, 0
        grid[r - 2][1], grid[r - 2][2] = 0, 1
        grid[1][c - 3], grid[1][c - 2] = 0, 1
        grid[2][c - 3], grid[2][c - 2] = 1, 0

    return grid


def example_layout_codes(rows: int = EXAMPLE_LAYOUT_ROWS, cols: int = EXAMPLE_LAYOUT_COLS,
                          include_diagonals: bool = True) -> list:
    """Returns a rows x cols grid of corner-combo codes (including '_pure_a'
    / '_pure_b') forming one static, self-consistent example terrain."""
    grid = example_corner_grid(rows, cols, include_diagonals)
    codes = []
    for y in range(rows):
        row_codes = []
        for x in range(cols):
            bits = (grid[y][x], grid[y][x + 1], grid[y + 1][x], grid[y + 1][x + 1])
            if bits == (0, 0, 0, 0):
                row_codes.append("_pure_a")
            elif bits == (1, 1, 1, 1):
                row_codes.append("_pure_b")
            else:
                row_codes.append(_BITS_TO_CODE[bits])
        codes.append(row_codes)
    return codes


CODE_LABELS = {
    "corner-tl": "Corner: top-left is B",
    "corner-tr": "Corner: top-right is B",
    "corner-bl": "Corner: bottom-left is B",
    "corner-br": "Corner: bottom-right is B",
    "edge-top": "Edge: top is B",
    "edge-bottom": "Edge: bottom is B",
    "edge-left": "Edge: left is B",
    "edge-right": "Edge: right is B",
    "inner-tl": "Inner corner: top-left is A",
    "inner-tr": "Inner corner: top-right is A",
    "inner-bl": "Inner corner: bottom-left is A",
    "inner-br": "Inner corner: bottom-right is A",
    "diag-tlbr": "Diagonal: TL+BR are B",
    "diag-trbl": "Diagonal: TR+BL are B",
    "island": "Island: lone A tile in B",
    "pond": "Pond: lone B tile in A",
    "_pure_a": "Pure A (base terrain)",
    "_pure_b": "Pure B (overlay terrain)",
    "_pure_c": "Pure C (transition terrain)",
}


_LEADING_INDEX_RE = re.compile(r"^\d+_+")


def strip_leading_index(stem: str) -> str:
    """Strips an existing 'NNN_' numeric prefix, e.g. '041_water' -> 'water'."""
    return _LEADING_INDEX_RE.sub("", stem) or stem


def sequential_filenames(start_index: int, overlay_stem: str, ext: str) -> dict:
    overlay_stem = strip_leading_index(overlay_stem)
    names = {}
    idx = int(start_index)
    for code in LEGACY_ORDER:
        key = "_pure_b" if code == "__PURE_B__" else code
        names[key] = f"{idx:03d}_{overlay_stem}{ext}"
        idx += 1
    return names


def descriptive_filenames(a_stem: str, b_stem: str, codes: list, ext: str,
                           c_stem: str = "transition") -> dict:
    names = {}
    for code in codes:
        if code == "_pure_a":
            names[code] = f"{a_stem}_full{ext}"
        elif code == "_pure_b":
            names[code] = f"{b_stem}_full{ext}"
        elif code == "_pure_c":
            names[code] = f"{c_stem}_full{ext}"
        else:
            names[code] = f"{a_stem}_{b_stem}_{code}{ext}"
    return names
