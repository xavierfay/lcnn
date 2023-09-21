import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import skimage.draw
from scipy.ndimage import zoom
from itertools import combinations
def to_int(x):
    return tuple(map(int, x))
def save_heatmap(prefix, image, lines):
    im_rescale = (1024, 1024)
    heatmap_scale = (256, 256)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    jmap = np.zeros((1,) + heatmap_scale, dtype=np.float32)
    joff = np.zeros((1, 2) + heatmap_scale, dtype=np.float32)
    lmap = np.zeros(heatmap_scale, dtype=np.float32)

    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    lines = lines[:, :, ::-1]

    junc = []
    jids = {}

    def jid(jun):
        jun = tuple(jun[:2])
        if jun in jids:
            return jids[jun]
        jids[jun] = len(junc)
        junc.append(np.array(jun + (0,)))
        return len(junc) - 1

    lnid = []
    lpos, lneg = [], []
    for v0, v1 in lines:
        lnid.append((jid(v0), jid(v1)))
        lpos.append([junc[jid(v0)], junc[jid(v1)]])

        vint0, vint1 = to_int(v0), to_int(v1)
        jmap[0][vint0] = 1
        jmap[0][vint1] = 1
        rr, cc, value = skimage.draw.line_aa(*to_int(v0), *to_int(v1))
        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

    for v in junc:
        vint = to_int(v[:2])
        joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5

    llmap = zoom(lmap, [0.5, 0.5])
    lineset = set([frozenset(l) for l in lnid])
    for i0, i1 in combinations(range(len(junc)), 2):
        if frozenset([i0, i1]) not in lineset:
            v0, v1 = junc[i0], junc[i1]
            vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)
            rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
            lneg.append([v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))])

    assert len(lneg) != 0
    lneg.sort(key=lambda l: -l[-1])

    junc = np.array(junc, dtype=np.float32)
    Lpos = np.array(lnid, dtype=int)
    Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=int)
    lpos = np.array(lpos, dtype=np.float32)
    lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)

    print('joff', joff)

    image = cv2.resize(image, im_rescale)
    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]    Junction heat map
        joff=joff,  # [J, 2, H, W] Junction offset within each pixel
        lmap=lmap,  # [H, W]       Line heat map with anti-aliasing
        junc=junc,  # [Na, 3]      Junction coordinate
        Lpos=Lpos,  # [M, 2]       Positive lines represented with junction indices
        Lneg=Lneg,  # [M, 2]       Negative lines represented with junction indices
        lpos=lpos,  # [Np, 2, 3]   Positive lines represented with junction coordinates
        lneg=lneg,  # [Nn, 2, 3]   Negative lines represented with junction coordinates
    )
    cv2.imwrite(f"{prefix}.png", image)



symbol_path = 'C:\\Users\\xavier\\Documents\\Thesis\\Demo_PIDs\\DigitizePID_Dataset-20230516T150124Z-001\\DigitizePID_Dataset\\0\\0_symbols.npy'
image = cv2.imread('C:\\Users\\xavier\\Documents\\GitHub\\lcnn\\data\\dpid_raw\\images\\0.png')
symbols = np.load(symbol_path, allow_pickle=True)
print(symbols )

fig, ax = plt.subplots(1)
ax.imshow(image)
center_points = [(0.5 * (x1 + x2), 0.5 * (y1 + y2)) for _, [x1, y1, x2, y2], _ in symbols]
tensor_sym = torch.Tensor(center_points)
zeros = torch.zeros(tensor_sym.size(0), 1)
tensor_sym = torch.cat((tensor_sym, zeros), dim=1)

print(tensor_sym)

lines = np.array(center_points)[:, :2]
lines = np.array([lines[:-1], lines[1:]]).transpose(1, 0, 2)
save_heatmap("output_prefix", image, lines)

#torch.save(tensor_sym, '/data/tensor_sym.pt')

for entry in symbols:
    symbol, (x1, y1, x2, y2), _ = entry
    center_points = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
    x_coords, y_coords = center_points
    plt.scatter(x_coords, y_coords, color='blue', label='Center Points')
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1, symbol, color='blue', fontsize=12, verticalalignment='top')

plt.show()



