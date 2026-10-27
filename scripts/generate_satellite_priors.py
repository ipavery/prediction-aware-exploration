#!/usr/bin/env python3
"""Create simulated building-footprint priors from collected LaMa ground truth.

Requires NumPy and Pillow (no model, GPU, OpenCV, or SciPy required).

Example on the cluster:
    python generate_satellite_priors.py \
        --gt-dir /home/ipa28/nobackup/autodelete/lama_map_pairs/train_3/global_gt \
        --save-outline

By default, writes matching single-channel PNGs to sibling global_satellite/,
plus a metadata JSONL and a preview. Source GT and observation files are read
only. Encoding: source GT 255=occupied, 0=free; prior 255=building footprint,
0=outside. A footprint is an extent prior, NEVER an occupancy/free-space label.

Extraction closes small wall gaps, fills enclosed regions, and removes tiny
components. This is a floor-plan silhouette heuristic, not satellite truth.
Check previews: open exterior walls, courtyards, detached structures, and upper
floors can make the approximation inaccurate. --method convex-hull is a coarse
alternative that fills concavities. --close-radius is in collected-map pixels.

Perturbations affect ONLY the prior, around its centroid. GT contents determine
the random seed, so repeated observations of the same GT get a consistent prior
and adding files does not change existing priors. --seed-by filename instead
creates independent perturbations for differently named pairs of the same GT.
Fresh jitter and whole-prior dropout during training are still recommended;
this script makes one reproducible prior per exported pair. No training-model
integration is performed by this utility.
"""

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

RESAMPLING = getattr(Image, 'Resampling', Image)
TRANSFORM = getattr(Image, 'Transform', Image)


def mask_image(mask):
    return Image.fromarray(mask.astype(np.uint8) * 255)


def morph(mask, radius, operation):
    if radius == 0:
        return mask.copy()
    border = radius + 1
    padded = mask_image(np.pad(mask, border, constant_values=False))
    size = 2 * radius + 1
    if operation == 'close':
        filtered = padded.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    elif operation == 'dilate':
        filtered = padded.filter(ImageFilter.MaxFilter(size))
    elif operation == 'erode':
        filtered = padded.filter(ImageFilter.MinFilter(size))
    else:
        raise ValueError('Unknown morphology operation: ' + operation)
    return np.asarray(filtered)[border:-border, border:-border] > 0


def fill_holes(mask):
    # Add an exterior margin even if a building touches the image boundary.
    # fromarray can create a read-only shared buffer; floodfill needs a copy.
    padded = mask_image(np.pad(mask, 1, constant_values=False)).copy()
    ImageDraw.floodfill(padded, (0, 0), 128, thresh=0)
    return (np.asarray(padded)[1:-1, 1:-1] != 128)


def remove_small_components(mask, minimum):
    if minimum <= 1:
        return mask.copy()
    height, width = mask.shape
    seen = np.zeros_like(mask)
    output = np.zeros_like(mask)
    for flat in np.flatnonzero(mask):
        row, col = divmod(int(flat), width)
        if seen[row, col]:
            continue
        queue = deque([(row, col)])
        seen[row, col] = True
        component = []
        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        if len(component) >= minimum:
            yy, xx = zip(*component)
            # Odd-size block reduction can add an isolated full-length occupied
            # row/column. Such lines are padding artifacts, not footprints.
            if max(yy) - min(yy) >= 2 and max(xx) - min(xx) >= 2:
                output[np.asarray(yy), np.asarray(xx)] = True
    return output


def convex_hull_mask(mask):
    # Reduce the hull input to the left/right extreme of each occupied row.
    points = []
    for y in np.flatnonzero(mask.any(axis=1)):
        xs = np.flatnonzero(mask[y])
        points.extend([(int(xs[0]), int(y)), (int(xs[-1]), int(y))])
    points = sorted(set(points))
    if len(points) < 3:
        raise ValueError('Not enough occupied points for a footprint hull')
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    output = Image.new('L', (mask.shape[1], mask.shape[0]))
    ImageDraw.Draw(output).polygon(hull, fill=255)
    return np.asarray(output) > 0


def extract_footprint(walls, close_radius=3, min_component=64, method='closed-envelope'):
    if not walls.any():
        raise ValueError('Ground truth contains no occupied cells')
    if method == 'convex-hull':
        footprint = convex_hull_mask(walls)
    else:
        footprint = fill_holes(morph(walls, close_radius, 'close'))
    footprint = remove_small_components(footprint, min_component)
    if not footprint.any():
        raise ValueError('No footprint survived filtering; inspect GT and --min-component')
    return footprint


def perturb_footprint(footprint, rng, shift_px=3, rotate_deg=2.0, scale_jitter=0.02,
                      boundary_jitter=1):
    dx, dy = (int(v) for v in rng.integers(-shift_px, shift_px + 1, size=2))
    angle = float(rng.uniform(-rotate_deg, rotate_deg))
    scale = float(rng.uniform(1 - scale_jitter, 1 + scale_jitter))
    bias = int(rng.integers(-boundary_jitter, boundary_jitter + 1))
    adjusted = footprint.copy()
    if bias:
        adjusted = morph(adjusted, abs(bias), 'dilate' if bias > 0 else 'erode')
    y, x = np.nonzero(footprint)
    cx, cy = float(x.mean()), float(y.mean())
    theta = math.radians(angle)
    a, b = math.cos(theta) / scale, math.sin(theta) / scale
    d, e = -b, a
    coefficients = (a, b, cx - a * (cx + dx) - b * (cy + dy),
                    d, e, cy - d * (cx + dx) - e * (cy + dy))
    image = mask_image(adjusted)
    transformed = image.transform(image.size, TRANSFORM.AFFINE, coefficients,
                                  resample=RESAMPLING.NEAREST, fillcolor=0)
    prior = np.asarray(transformed) > 0
    metadata = dict(dx_px=dx, dy_px=dy, angle_deg=angle, scale=scale,
                    boundary_bias_px=bias, center_xy=[cx, cy],
                    prior_pixels=int(prior.sum()))
    return prior, metadata


def outline(mask):
    return mask & ~morph(mask, 1, 'erode')


def sample_rng(name, seed):
    digest = hashlib.sha256((str(seed) + ':' + name).encode('utf-8')).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'big'))


def read_gt(path):
    with Image.open(path) as source:
        rgb = np.asarray(source.convert('RGB'))
    if not np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) or not np.array_equal(rgb[:, :, 0], rgb[:, :, 2]):
        raise ValueError('Expected replicated grayscale ground truth: ' + str(path))
    gt = rgb[:, :, 0]
    if not np.isin(gt, [0, 255]).all():
        raise ValueError('Expected binary collected GT (0=free, 255=occupied): ' + str(path))
    return gt, gt == 255


def font(size):
    for path in ('C:/Windows/Fonts/segoeui.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def preview(samples, destination):
    cell_w, cell_h = 300, 330
    canvas = Image.new('RGB', (4 * cell_w + 40, len(samples) * cell_h + 90), '#edf1f5')
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 14), 'Simulated satellite footprint: extent, not indoor occupancy', font=font(23), fill='#172738')
    titles = ('GT (display: dark walls)', 'Clean filled footprint', 'Perturbed filled prior', 'GT + perturbed outline')
    for row, (name, gt, clean, prior, metadata) in enumerate(samples):
        edge = outline(prior)
        base = np.repeat((255 - gt)[:, :, None], 3, axis=2)
        fill = np.full(base.shape, 242, dtype=np.uint8)
        fill[clean] = (17, 126, 139)
        disturbed = np.full(base.shape, 242, dtype=np.uint8)
        disturbed[prior] = (17, 126, 139)
        overlay = base.copy()
        overlay[edge] = (226, 103, 37)
        arrays = (base, fill, disturbed, overlay)
        for col, array in enumerate(arrays):
            x, y = 20 + col * cell_w, 64 + row * cell_h
            draw.rounded_rectangle((x, y, x + cell_w - 10, y + cell_h - 10), radius=8, fill='white')
            draw.text((x + 10, y + 8), titles[col], font=font(17), fill='#172738')
            img = Image.fromarray(array)
            factor = min(265 / img.width, 250 / img.height)
            img = img.resize((max(1, round(img.width * factor)), max(1, round(img.height * factor))), RESAMPLING.NEAREST)
            canvas.paste(img, (x + (cell_w - 10 - img.width) // 2, y + 40 + (250 - img.height) // 2))
            label = name[:33] if col == 0 else ''
            if col == 2:
                label = 'shift ({}, {}), rotation {:.1f} deg'.format(metadata['dx_px'], metadata['dy_px'], metadata['angle_deg'])
            if col == 3:
                label = 'Diagnostic overlay; not model input'
            draw.text((x + 9, y + 294), label, font=font(13), fill='#425268')
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gt-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, help='Default: sibling global_satellite')
    parser.add_argument('--method', choices=('closed-envelope', 'convex-hull'), default='closed-envelope')
    parser.add_argument('--close-radius', type=int, default=3)
    parser.add_argument('--min-component', type=int, default=64)
    parser.add_argument('--shift-px', type=int, default=3)
    parser.add_argument('--rotate-deg', type=float, default=2.0)
    parser.add_argument('--scale-jitter', type=float, default=0.02)
    parser.add_argument('--boundary-jitter', type=int, default=1)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--seed-by', choices=('gt', 'filename'), default='gt')
    parser.add_argument('--save-outline', action='store_true')
    parser.add_argument('--preview-count', type=int, default=4)
    args = parser.parse_args()
    for name in ('close_radius', 'shift_px', 'rotate_deg', 'boundary_jitter', 'preview_count'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error('--' + name.replace('_', '-') + ' must be finite and nonnegative')
    if not math.isfinite(args.scale_jitter) or not 0 <= args.scale_jitter < 1:
        parser.error('--scale-jitter must be finite and in [0, 1)')
    if args.min_component < 1:
        parser.error('--min-component must be positive')
    gt_dir = args.gt_dir.expanduser().resolve()
    paths = sorted(gt_dir.glob('*.png'))
    if not paths:
        parser.error('No PNG ground truths found under ' + str(gt_dir))
    output_dir = (args.output_dir or gt_dir.parent / 'global_satellite').expanduser().resolve()
    if output_dir == gt_dir or output_dir == gt_dir.parent / 'global_obs':
        parser.error('Output must be separate from GT and observations')
    outline_dir = output_dir.parent / (output_dir.name + '_outline')
    preview_path = output_dir.parent / (output_dir.name + '_preview.png')
    targets = [output_dir / p.name for p in paths] + [output_dir / 'manifest.jsonl']
    if args.preview_count:
        targets.append(preview_path)
    if args.save_outline:
        targets.extend(outline_dir / p.name for p in paths)
    existing = [p for p in targets if p.exists()]
    if existing:
        parser.error('Output already exists; choose a fresh --output-dir: ' + str(existing[0]))
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_outline:
        outline_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    config = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
    with (output_dir / 'manifest.jsonl').open('w', encoding='utf-8') as manifest:
        for path in paths:
            gt, walls = read_gt(path)
            clean = extract_footprint(walls, args.close_radius, args.min_component, args.method)
            gt_content_key = hashlib.sha256(str(gt.shape).encode('ascii') + gt.tobytes()).hexdigest()
            seed_key = gt_content_key if args.seed_by == 'gt' else path.stem
            prior, metadata = perturb_footprint(clean, sample_rng(seed_key, args.seed),
                args.shift_px, args.rotate_deg, args.scale_jitter, args.boundary_jitter)
            if not prior.any():
                raise ValueError('Perturbation removed entire footprint: ' + str(path))
            warnings = []
            filled_pixels = int((clean & ~walls).sum())
            if filled_pixels < int(walls.sum()):
                warnings.append('Little enclosed interior recovered; check exterior gaps and footprint extraction')
            if prior[0].any() or prior[-1].any() or prior[:, 0].any() or prior[:, -1].any():
                warnings.append('Prior touches image edge; footprint may be clipped')
            mask_image(prior).save(output_dir / path.name)
            if args.save_outline:
                mask_image(outline(prior)).save(outline_dir / path.name)
            metadata.update(name=path.name, gt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                height=int(gt.shape[0]), width=int(gt.shape[1]),
                clean_footprint_pixels=int(clean.sum()), filled_interior_pixels=filled_pixels,
                seed_key=seed_key, config=config, warnings=warnings)
            manifest.write(json.dumps(metadata) + '\n')
            manifest.flush()
            for warning in warnings:
                print('REVIEW ' + path.name + ': ' + warning)
            if len(samples) < args.preview_count:
                samples.append((path.stem, gt, clean, prior, metadata))
    if samples:
        preview(samples, preview_path)
    print('Saved {} filled footprint priors to {}'.format(len(paths), output_dir))
    if args.save_outline:
        print('Matching perturbed outlines: ' + str(outline_dir))


if __name__ == '__main__':
    main()
