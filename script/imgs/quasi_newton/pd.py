import matplotlib.pyplot as plt
import numpy as np

from qnlab.util.doc_paths import doc_imgs_dir

OUTPUT_DIR = doc_imgs_dir("quasi_newton")

H_pos = np.array([[2.0, 0.4], [0.4, 1.2]])
H_ind = np.array([[2.5, 0.0], [0.0, -2.5]])
H_neg = np.array([[-2.0, -0.3], [-0.3, -1.0]])
hessians = [
    ("Positive definite", H_pos),
    ("Indefinite", H_ind),
    ("Negative definite", H_neg),
]


def compute_surface_data(H):
    b = np.zeros(2)  # -H.dot([0,0]) = [0,0]
    c = 0.0
    lim = 2.5
    n = 50
    xs = np.linspace(-lim, +lim, n)
    ys = np.linspace(-lim, +lim, n)
    X, Y = np.meshgrid(xs, ys)
    XY = np.stack([X, Y], axis=-1)
    Z = 0.5 * np.einsum("...i,ij,...j", XY, H, XY) + np.einsum("...i,i", XY, b) + c
    return X, Y, Z, b, c


def plot_surfaces_and_save():
    fig = plt.figure(figsize=(10, 5))

    # Calculate global z-range for consistent colormap
    all_surfaces = [compute_surface_data(H) for _, H in hessians]
    z_min = min(Z.min() for _, _, Z, _, _ in all_surfaces)
    z_max = max(Z.max() for _, _, Z, _, _ in all_surfaces)

    for i, (title, H) in enumerate(hessians):
        X, Y, Z, _b, _c = all_surfaces[i]

        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_surface(
            X,
            Y,
            Z,
            linewidth=0,
            antialiased=True,
            cmap="viridis",
            vmin=z_min,
            vmax=z_max,
            zsort="min",
        )

        # 軸を消す
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])  # type: ignore
        ax.grid(False)
        ax.axis("off")
        # enlarge title (2.5x of previous 12 -> 30)
        ax.set_title(title, pad=12, fontsize=30)
        ax.view_init(elev=30, azim=-60)

    # Reduce spacing between subplots as much as practical
    plt.tight_layout(pad=0.0)
    plt.subplots_adjust(wspace=0.01)
    plt.savefig(OUTPUT_DIR / "pd.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_surfaces_and_save()
