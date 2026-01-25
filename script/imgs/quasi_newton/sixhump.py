from __future__ import annotations

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np

from qnlab.problem import SixHumpProblem
from qnlab.solver.qn import qn
from qnlab.update.bfgs import compute_BH
from qnlab.util.callback import Callback
from qnlab.util.doc_paths import doc_imgs_dir
from qnlab.util.iteration_data import IterationData
from qnlab.util.memory_interface import QuasiNewtonMemory
from qnlab.util.method import Method

OUTPUT_DIR = doc_imgs_dir("quasi_newton")

GRID_SIZE = 150
X_RANGE = (-2, 0.5)
Y_RANGE = (-1, 1)
X_LEN = X_RANGE[1] - X_RANGE[0]
Y_LEN = Y_RANGE[1] - Y_RANGE[0]
FIGURE_SIZE = (6, 8)
VIEW_ELEVATION = 40
VIEW_AZIMUTH = 30
PATH_COLOR = "crimson"
PATH_MARKER_SIZE = 3
PATH_LINE_WIDTH = 2
CROP_RECT = (30, 30, 410, 370)
ITER_GRID_SIZE = 50
ITER_REGION_RADIUS = 0.5
QUADRATIC_ALPHA = 0.9
WIREFRAME_LINEWIDTH = 0.5
WIREFRAME_ALPHA = 0.3


def build_scene(
    xg, yg, zg, surface_alpha: float, add_path: bool, path=None, path_z=None
):
    fig = plt.figure(figsize=FIGURE_SIZE)
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
        if path is None or path_z is None:
            raise ValueError("path and path_z must be provided when add_path is True")
        ax.plot(
            path[:, 0],
            path[:, 1],
            path_z,
            color=PATH_COLOR,
            marker="o",
            markersize=PATH_MARKER_SIZE,
            linewidth=PATH_LINE_WIDTH,
        )
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


def save_layer(fig, target, transparent: bool):
    fmt = target.suffix[1:]
    save_kwargs = {"format": fmt, "bbox_inches": "tight", "pad_inches": 0}
    if transparent:
        save_kwargs.update(facecolor="none", edgecolor="none", transparent=True)
    fig.savefig(target, **save_kwargs)
    plt.close(fig)


def create_surface_grid(prob):
    x = np.linspace(*X_RANGE, GRID_SIZE)
    y = np.linspace(*Y_RANGE, GRID_SIZE)
    xg, yg = np.meshgrid(x, y)
    zg = prob.f(np.array([xg, yg]), count=False)
    return xg, yg, zg


def run_optimization(prob):
    callback = Callback(gnorm_order=2, save_xs=True)
    method = Method(base="SciPy", scipy_method="L-BFGS-B")
    _, f_opt, x_opt = qn(
        prob, method, options={"maxiter": 9}, callback=callback, verbose=False
    )
    if not callback.xs or not np.allclose(callback.xs[-1], x_opt):
        callback.callback(
            prob, x_opt, prob.f(x_opt, count=False), prob.g(x_opt, count=False)
        )
    print(len(callback.xs))
    return callback, x_opt


def generate_gif(
    prob, path, path_z, output_dir, xg_global, yg_global, zg_global, callback
):
    gradients = [prob.g(x, count=False) for x in callback.xs]
    lm_list = [
        IterationData(
            s=callback.xs[k] - callback.xs[k - 1], y=gradients[k] - gradients[k - 1]
        )
        for k in range(1, len(callback.xs))
    ]

    print("\nGenerating iteration visualizations with L-BFGS approximation...")

    init_z_lim = (0.0, 0.0)

    for k in range(len(path)):
        path_up_to_k = path[: k + 1]
        path_z_up_to_k = path_z[: k + 1]
        x_k = path[k]
        f_k = callback.fxs[k]
        g_k = gradients[k]

        method = Method(base="SciPy", scipy_method="L-BFGS-B")
        lm = QuasiNewtonMemory(g=g_k, maxlen=100, method=method)
        for item in lm_list[:k]:
            lm._deque.append(item)

        B_k = compute_BH(len(x_k), lm)[0] if len(lm) > 0 else np.eye(len(x_k))

        fig = plt.figure(figsize=FIGURE_SIZE)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(
            xg_global,
            yg_global,
            zg_global,
            linewidth=WIREFRAME_LINEWIDTH,
            alpha=WIREFRAME_ALPHA,
        )

        x_min = max(x_k[0] - ITER_REGION_RADIUS, X_RANGE[0] + 0.05 * X_LEN)
        x_max = min(x_k[0] + ITER_REGION_RADIUS, X_RANGE[1] - 0.05 * X_LEN)
        y_min = max(x_k[1] - ITER_REGION_RADIUS, Y_RANGE[0] + 0.05 * Y_LEN)
        y_max = min(x_k[1] + ITER_REGION_RADIUS, Y_RANGE[1] - 0.05 * Y_LEN)
        xg_local, yg_local = np.meshgrid(
            np.linspace(x_min, x_max, ITER_GRID_SIZE),
            np.linspace(y_min, y_max, ITER_GRID_SIZE),
        )

        dx = xg_local - x_k[0]
        dy = yg_local - x_k[1]
        zg_quad = (
            f_k
            + g_k[0] * dx
            + g_k[1] * dy
            + 0.5 * (B_k[0, 0] * dx**2 + 2 * B_k[0, 1] * dx * dy + B_k[1, 1] * dy**2)
        )

        ax.plot_surface(
            xg_local,
            yg_local,
            zg_quad,
            cmap=plt.colormaps.get_cmap("viridis"),
            linewidth=0,
            alpha=QUADRATIC_ALPHA,
        )

        if len(path_up_to_k) > 0:
            ax.plot(
                path_up_to_k[:, 0],
                path_up_to_k[:, 1],
                path_z_up_to_k,
                color=PATH_COLOR,
                marker="o",
                markersize=PATH_MARKER_SIZE,
                linewidth=PATH_LINE_WIDTH,
            )

        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])  # type: ignore
        if k == 0:
            init_z_lim = ax.get_zlim()
        ax.set_zlim(init_z_lim)
        ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
        fig.subplots_adjust(0, 0, 1, 1)

        frame_path = output_dir / f"sixhump_iteration_{k:03d}.pdf"
        fig.savefig(frame_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def generate_output(prob, path, path_z, xg, yg, zg, callback, output_dir):
    pdf_transparent = output_dir / "sixhump_path_layer.pdf"
    pdf_opaque = output_dir / "sixhump_surface_base.pdf"
    final_pdf = output_dir / "sixhump.pdf"

    save_layer(
        build_scene(
            xg, yg, zg, surface_alpha=0.0, add_path=True, path=path, path_z=path_z
        ),
        pdf_transparent,
        transparent=True,
    )
    save_layer(
        build_scene(xg, yg, zg, surface_alpha=1.0, add_path=False),
        pdf_opaque,
        transparent=False,
    )
    # Compose layered PDF using PyMuPDF
    base_doc = fitz.open(pdf_opaque)
    layer_doc = fitz.open(pdf_transparent)
    final_doc = fitz.open()
    final_doc.insert_pdf(base_doc)
    page = final_doc[0]
    rect = page.rect
    page.show_pdf_page(rect, layer_doc, pno=0)
    final_doc.save(final_pdf)
    final_doc.close()
    base_doc.close()
    layer_doc.close()
    pdf_transparent.unlink(missing_ok=True)
    pdf_opaque.unlink(missing_ok=True)

    print(f"Layered visualization saved to {final_pdf}")


def main():
    prob = SixHumpProblem()
    xg, yg, zg = create_surface_grid(prob)
    callback, x_opt = run_optimization(prob)
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = np.array(callback.xs)
    path_z = prob.f(path.T, count=False)
    generate_output(prob, path, path_z, xg, yg, zg, callback, output_dir)
    generate_gif(prob, path, path_z, output_dir, xg, yg, zg, callback)


if __name__ == "__main__":
    main()
