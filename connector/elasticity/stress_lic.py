#!/usr/bin/env python
"""
Szymon Rusinkiewicz
Princeton University

stress_lic.py
Visualization of stress tensors with LIC
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def valid(mask, pos):
    """Determine whether pos is a valid position"""
    x = math.floor(pos[0])
    if (x < 0) or (x >= mask.shape[0] - 1):
        return False
    y = math.floor(pos[1])
    if (y < 0) or (y >= mask.shape[1] - 1):
        return False
    return (mask[x, y] and mask[x, y + 1] and mask[x + 1, y]
            and mask[x + 1, y + 1])


def normalized(v):
    """Return normalized (unit-length) version of vector v."""
    return v * (1.0 / np.linalg.norm(v))


def lerp(field, pos):
    """Bilinearly interpolate the value of field at pos.  No error checking."""
    x0 = math.floor(pos[0])
    x1 = x0 + 1
    ax1 = pos[0] - x0
    ax0 = 1.0 - ax1
    y0 = math.floor(pos[1])
    y1 = y0 + 1
    ay1 = pos[1] - y0
    ay0 = 1.0 - ay1
    ret = ax0 * (ay0 * field[x0, y0] + ay1 * field[x0, y1])
    ret += ax1 * (ay0 * field[x1, y0] + ay1 * field[x1, y1])
    return ret


def splat(pos, val, img):
    """Bilinearly splat val into img at pos.  No error checking."""
    x = pos[0]
    x0 = math.floor(x)
    x1 = x0 + 1
    ax1 = x - x0
    ax0 = 1.0 - ax1
    y = pos[1]
    y0 = math.floor(y)
    y1 = y0 + 1
    ay1 = y - y0
    ay0 = 1.0 - ay1
    img[x0, y0] += (ax0*ay0) * val
    img[x0, y1] += (ax0*ay1) * val
    img[x1, y0] += (ax1*ay0) * val
    img[x1, y1] += (ax1*ay1) * val


def draw_path(tensors, mask, pos0, dir0, color0, scale, tmult, dirmult, nsteps,
              skip_first, img):
    """Walk and draw a path starting at pos0 in direction dir0"""
    pos = pos0.copy()
    dir = dir0.copy()
    pos_color = color0.copy()
    pos_color[1] *= 0.5
    pos_color[2] *= 0.5
    neg_color = color0.copy()
    neg_color[0] *= 0.5
    neg_color[1] *= 0.5
    hpos = 1.0 / scale
    hdir = hpos * dirmult

    for _ in range(nsteps):
        if not valid(mask, pos):
            return
        t = tmult * (lerp(tensors, pos) @ dir)
        val = dir @ t
        color = color0.copy()
        if val >= 0.0:
            val = min(max(0.95 - val, 0), 1)
            color[1] *= val
            color[2] *= val
        else:
            val = min(max(0.95 + val, 0), 1)
            color[0] *= val
            color[1] *= val
            t = -t
        if not skip_first:
            splat(pos * scale, color, img)

        dir = normalized(dir + hdir*t)
        pos += hpos * dir


def draw_lic(stress, mask, scale=20, spacing=2, flowlen=8, dirmult=50):
    """Draw the LIC visualization of an input stress .npz, output to a .png"""
    nx = stress.shape[0]
    ny = stress.shape[1]
    tmult = 0.9 / max(abs(stress.max()), abs(stress.min()))

    iscale = 1.0 / scale
    img = np.full(((nx-1) * scale + 1, (ny-1) * scale + 1, 4), 0.5)
    img[:, :, 3] = 1
    nsteps = flowlen * scale // 2

    for x in tqdm(range(0, nx * scale, spacing)):
        for y in range(0, ny * scale, spacing):
            xx = iscale * (x + np.random.rand() * spacing)
            yy = iscale * (y + np.random.rand() * spacing)
            pos = np.array([xx, yy])
            if not valid(mask, pos):
                continue

            angle = np.random.rand() * 2 * np.pi
            dir = np.array([math.cos(angle), math.sin(angle)])

            # c = np.random.rand()
            c = 0.2
            if np.random.rand() > 0.5:
                c = 1.0
            color = np.array([c, c, c, 1])

            draw_path(stress, mask, pos, dir, color, scale, tmult, dirmult,
                      nsteps, False, img)
            draw_path(stress, mask, pos, -dir, color, scale, tmult, dirmult,
                      nsteps, True, img)

    img = img[:, :, 0:3] / img[:, :, 3:]
    img = img.transpose((1, 0, 2))
    img = np.flip(img, 0)

    plt.figure(dpi=500)
    plt.imshow(img)
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.show()
