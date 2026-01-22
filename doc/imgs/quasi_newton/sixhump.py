# Six-hump Camelback function
# The objective is
# f(x, y) = (4 - 2.1x^2 + \frac{x^4}{3}) x^2 + xy + (-4 + 4y^2) y^2.
# We visualize the surface and the L-BFGS iterates starting from $(x_0, y_0) = (-1.8, -0.8)$.

import os
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import minimize


def sixhump_with_grad(x: np.ndarray):
    x0, x1 = x
    term1 = (4 - 2.1 * x0**2 + (x0**4) / 3.0) * x0**2
    term2 = x0 * x1
    term3 = (-4 + 4 * x1**2) * x1**2
    value = term1 + term2 + term3
    grad_x0 = 8 * x0 - 8.4 * x0**3 + 2 * x0**5 + x1
    grad_x1 = x0 - 8 * x1 + 16 * x1**3
    return value, np.array([grad_x0, grad_x1])


def sixhump_value(x: np.ndarray):
    return sixhump_with_grad(x)[0]


def sixhump_grad(x: np.ndarray):
    return sixhump_with_grad(x)[1]


def strip_axes(ax):
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def build_scene(surface_alpha: float, add_path: bool):
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xg,
        yg,
        zg,
        cmap=plt.colormaps.get_cmap("viridis"),
        linewidth=0,
        alpha=surface_alpha,
    )
    if add_path:
        ax.plot(
            path[:, 0],
            path[:, 1],
            path_z,
            color="crimson",
            marker="o",
            markersize=3,
            linewidth=2,
        )
    strip_axes(ax)
    ax.view_init(elev=40, azim=30)
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


def save_layer(surface_alpha: float, add_path: bool, target: Path, transparent: bool):
    fig = build_scene(surface_alpha, add_path)
    save_kwargs = {"format": "pdf", "bbox_inches": "tight", "pad_inches": 0}
    if transparent:
        save_kwargs.update(facecolor="none", edgecolor="none", transparent=True)
    fig.savefig(target, **save_kwargs)
    plt.close(fig)


def crop_page_to_content(page: fitz.Page) -> None:
    x0 = 30
    y0 = 30
    x1 = 410
    y1 = 370
    crop_rect = fitz.Rect(x0, y0, x1, y1)
    page.set_cropbox(crop_rect)


def merge_layers(
    layer_path: Path,
    base_path: Path,
    out_pdf: Path,
):
    doc_layer = fitz.open(layer_path)
    doc_base = fitz.open(base_path)
    output_doc = fitz.open()
    try:
        output_doc.insert_pdf(doc_base, from_page=0, to_page=0)
        output_page = output_doc[0]
        output_page.show_pdf_page(output_page.rect, doc_layer, 0, overlay=True)
        crop_page_to_content(output_page)
        output_doc.save(out_pdf, garbage=4, deflate=True, clean=True)
    finally:
        doc_layer.close()
        doc_base.close()
        output_doc.close()


# Grid for surface
x = np.linspace(-2, 0.5, 150)
y = np.linspace(-1, 1, 150)
xg, yg = np.meshgrid(x, y)
zg = sixhump_value(np.array([xg, yg]))

# L-BFGS-B from starting point (-1.8, -0.8)
x0 = np.array([-1.8, -0.8], dtype=float)
val0, grad0 = sixhump_with_grad(x0)
logs = [
    {
        "iter": -1,
        "x": float(x0[0]),
        "y": float(x0[1]),
        "f": float(val0),
        "grad_norm": float(np.linalg.norm(grad0)),
    }
]


def callback(xk: np.ndarray):
    val, grad = sixhump_with_grad(xk)
    logs.append(
        {
            "iter": len(logs),
            "x": float(xk[0]),
            "y": float(xk[1]),
            "f": float(val),
            "grad_norm": float(np.linalg.norm(grad)),
        }
    )


result = minimize(
    sixhump_value,
    x0=x0,
    method="L-BFGS-B",
    jac=sixhump_grad,
    callback=callback,
)

# Add final point to logs if not captured
val_final, grad_final = sixhump_with_grad(result.x)
if (
    not logs
    or logs[-1]["x"] != float(result.x[0])
    or logs[-1]["y"] != float(result.x[1])
):
    logs.append(
        {
            "iter": len(logs),
            "x": float(result.x[0]),
            "y": float(result.x[1]),
            "f": float(val_final),
            "grad_norm": float(np.linalg.norm(grad_final)),
        }
    )

print("Optimization log (iteration, x, y, f, ||grad||):")
for entry in logs:
    print(
        f"{entry['iter']:3d}: x={entry['x']:+.4f}, y={entry['y']:+.4f}, "
        f"f={entry['f']:+.6f}, grad_norm={entry['grad_norm']:.3e}"
    )

# Trajectory arrays
path = np.array([[e["x"], e["y"]] for e in logs])
path_z = sixhump_value(path.T)

# Output paths
output_dir = Path(os.path.dirname(__file__))
output_dir.mkdir(parents=True, exist_ok=True)
pdf_transparent = output_dir / "sixhump_path_layer.pdf"
pdf_opaque = output_dir / "sixhump_surface_base.pdf"
final_pdf = output_dir / "sixhump.pdf"

# 1. Transparent path layer (surface hidden)
save_layer(surface_alpha=0.0, add_path=True, target=pdf_transparent, transparent=True)

# 2. Opaque surface without path (base layer)
save_layer(surface_alpha=1.0, add_path=False, target=pdf_opaque, transparent=False)

# 3. Merge layers and export PNG
merge_layers(layer_path=pdf_transparent, base_path=pdf_opaque, out_pdf=final_pdf)
pdf_transparent.unlink(missing_ok=True)
pdf_opaque.unlink(missing_ok=True)

print(f"Layered visualization saved to {final_pdf}")
