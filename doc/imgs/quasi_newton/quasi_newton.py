import os
import tempfile

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update(
    {
        "text.usetex": True,  # LaTeX を使う
        "font.size": 18,  # 基本フォントサイズ（十分大きめ）
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 16,
    }
)


def f(XY):
    XY = np.asarray(XY)
    x = XY[..., 0]
    y = XY[..., 1]
    dx = x - 1.2
    dy = y + 1.2
    return dx**4 + 2.0 * dx**2 * dy**2 + dy**4 + 0.1 * dx**3 - 0.6 * dx**2


def grad_f(xy):
    xy = np.asarray(xy)
    x = xy[..., 0]
    y = xy[..., 1]
    dx = x - 1.2
    dy = y + 1.2
    gx = 4.0 * dx**3 + 4.0 * dx * dy**2 + 0.3 * dx**2 - 1.2 * dx
    gy = 4.0 * dy**3 + 4.0 * dy * dx**2
    return np.stack([gx, gy], axis=-1)


def hess_f(xy):
    xy = np.asarray(xy)
    x = xy[..., 0]
    y = xy[..., 1]
    dx = x - 1.2
    dy = y + 1.2

    H = np.empty(x.shape + (2, 2))
    H[..., 0, 0] = 12.0 * dx**2 + 4.0 * dy**2 + 0.6 * dx - 1.2
    H[..., 0, 1] = 8.0 * dx * dy
    H[..., 1, 0] = 8.0 * dx * dy
    H[..., 1, 1] = 12.0 * dy**2 + 4.0 * dx**2
    return H


x_k = np.array([-0.4, 0.3])
B_k = hess_f(x_k)
g_k = grad_f(x_k)

nx, ny = 100, 100
xs = np.linspace(-1.0, 1.0, nx)
ys = np.linspace(-1.0, 1.0, ny)
X, Y = np.meshgrid(xs, ys)
XY_grid = np.stack((X, Y), axis=-1)
Z = f(XY_grid)


# 二次モデル q(x) = f(x_k) + g_k^T (x-x_k) + 1/2 (x-x_k)^T B (x-x_k)
def quadratic_surface(B, xref, gref, X, Y):
    # X,Y : meshgrid arrays
    dx = X - xref[0]
    dy = Y - xref[1]
    # 1/2 [dx dy] B [dx; dy]
    qquad = 0.5 * (
        dx * (B[0, 0] * dx + B[0, 1] * dy) + dy * (B[1, 0] * dx + B[1, 1] * dy)
    )
    q = f(xref) + (gref[0] * dx + gref[1] * dy) + qquad
    return q


def beautify_ax(ax, transparent_panes=False):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # 軸面の透明度を設定
    if transparent_panes:
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.xaxis.line.set_alpha(0.0)
        ax.yaxis.line.set_alpha(0.0)
        ax.zaxis.line.set_alpha(0.0)
    else:
        # デフォルトの不透明度に戻す
        ax.xaxis.pane.set_alpha(1.0)
        ax.yaxis.pane.set_alpha(1.0)
        ax.zaxis.pane.set_alpha(1.0)
        ax.xaxis.line.set_alpha(1.0)
        ax.yaxis.line.set_alpha(1.0)
        ax.zaxis.line.set_alpha(1.0)


def draw_common_elements(ax, is_kp1=False):
    """共通の描画要素（点、テキスト、線）を描画する関数"""
    # x_k の点とテキスト
    if not is_kp1:
        ax.scatter([x_k[0]], [x_k[1]], [f(x_k)], s=100, color="red")
        ax.text(
            x_k[0],
            x_k[1] + 0.2,
            f(x_k),
            r"$x_k$",
            fontsize=45,
            color="red",
        )
    else:
        ax.scatter(
            [x_kp1[0]], [x_kp1[1]], [f(x_kp1)], s=100, color="yellow", marker="x"
        )
        ax.text(
            x_kp1[0],
            x_kp1[1] + 0.2,
            f(x_kp1),
            r"$x_{k+1}$",
            fontsize=45,
            color="yellow",
        )


def draw_surface_z(ax, alpha):
    """Z（元の関数）のサーフェスを描画"""
    ax.plot_surface(X, Y, Z, alpha=alpha, linewidth=0, rcount=50, ccount=50)


def draw_surface_quadratic(ax, Q, color, alpha):
    """二次関数サーフェスを描画"""
    ax.plot_surface(
        X, Y, Q, alpha=alpha, linewidth=0, color=color, rcount=50, ccount=50
    )
    if color == "tab:orange":
        ax.text2D(
            0.65,
            0.7,
            r"$m_{k}(x)$",
            transform=ax.transAxes,
            fontsize=40,
            color="tab:orange",
        )
    elif color == "tab:green":
        ax.text2D(
            0.6,
            0.7,
            r"$m_{k+1}(x)$",
            transform=ax.transAxes,
            fontsize=40,
            color="tab:green",
        )


def draw_step_elements(ax):
    """ステップ関連の要素（x_{k+1}, s_k）を描画"""
    # x_{k+1} の点とテキスト
    ax.scatter([x_kp1[0]], [x_kp1[1]], [f(x_kp1)], s=100, marker="x", color="yellow")
    ax.text(
        x_kp1[0], x_kp1[1] + 0.2, f(x_kp1), r"$x_{k+1}$", fontsize=45, color="yellow"
    )

    # ステップ s_k の線とテキスト
    ax.plot(
        [x_k[0], x_kp1[0]],
        [x_k[1], x_kp1[1]],
        [f(x_k), f(x_kp1)],
        linewidth=2,
        color="k",
    )
    mid = x_k + 0.5 * s_k
    ax.text(mid[0] - 0.2, mid[1] - 0.3, f(mid), r"$s_k$", fontsize=30)


def create_enhanced_figure(draw_func, filename):
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf1_path = os.path.join(tmpdir, "transparent.pdf")
        pdf2_path = os.path.join(tmpdir, "opaque.pdf")

        # 1. 透明なサーフェスでの描画（PDFとして保存）
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection="3d")
        draw_func(ax1, transparent_surfaces=True)
        beautify_ax(ax1, transparent_panes=True)
        plt.tight_layout()
        plt.savefig(
            pdf1_path,
            format="pdf",
            bbox_inches="tight",
            facecolor="none",
            edgecolor="none",
            transparent=True,
        )
        plt.close(fig1)

        # 2. 完全に不透明なサーフェスでの描画（PDFとして保存）
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111, projection="3d")
        draw_func(ax2, transparent_surfaces=False)
        beautify_ax(ax2, transparent_panes=False)
        plt.tight_layout()
        plt.savefig(pdf2_path, format="pdf", bbox_inches="tight")
        plt.close(fig2)

        # 3. PyMuPDFを使用してPDFを合成
        doc1 = fitz.open(pdf1_path)
        doc2 = fitz.open(pdf2_path)

        # 不透明な図をベースとして使用
        output_doc = fitz.open()
        output_doc.insert_pdf(doc2, from_page=0, to_page=0)

        # 透明な図を上に重ねる
        output_page = output_doc[0]

        # show_pdf_pageを使って透明な図を重ねる
        rect = output_page.rect
        output_page.show_pdf_page(rect, doc1, 0, overlay=True)

        # PDFとして保存（圧縮を有効化）
        output_doc.save(filename, garbage=4, deflate=True, clean=True)

        output_doc.close()
        doc1.close()
        doc2.close()


def draw_panel1(ax, transparent_surfaces=True):
    if transparent_surfaces:
        draw_surface_z(ax, alpha=0.0)
        draw_common_elements(ax)
        ax.text2D(
            0.65,
            0.7,
            r"$f(x)$",
            transform=ax.transAxes,
            fontsize=40,
            color="tab:blue",
        )
    else:
        draw_surface_z(ax, alpha=1.0)


create_enhanced_figure(draw_panel1, "quasi_newton_1.pdf")

Q_Bk = quadratic_surface(B_k, x_k, g_k, X, Y)

x_kp1 = x_k - np.linalg.solve(B_k, g_k)
s_k = x_kp1 - x_k
g_kp1 = grad_f(x_kp1)


def draw_panel2(ax, transparent_surfaces=True):
    if transparent_surfaces:
        draw_surface_z(ax, alpha=0.0)
        draw_surface_quadratic(ax, Q_Bk, "tab:orange", alpha=0.0)
        draw_common_elements(ax)
        draw_step_elements(ax)
    else:
        draw_surface_z(ax, alpha=1.0)
        draw_surface_quadratic(ax, Q_Bk, "tab:orange", alpha=0.7)


create_enhanced_figure(draw_panel2, "quasi_newton_2.pdf")

y_k = g_kp1 - g_k
den1 = float(s_k.T @ B_k @ s_k)
den2 = float(y_k.T @ s_k)

B_kp1 = B_k - (B_k @ np.outer(s_k, s_k) @ B_k) / den1 + np.outer(y_k, y_k) / den2
assert np.all(np.linalg.eigvals(B_kp1) > 0)

Q_Bkp1 = quadratic_surface(B_kp1, x_k, g_k, X, Y)


def draw_panel3(ax, transparent_surfaces=True):
    if transparent_surfaces:
        draw_surface_z(ax, alpha=0.0)
        draw_surface_quadratic(ax, Q_Bkp1, "tab:green", alpha=0.0)
        draw_common_elements(ax, is_kp1=True)
    else:
        draw_surface_z(ax, alpha=1.0)
        draw_surface_quadratic(ax, Q_Bkp1, "tab:green", alpha=0.7)


create_enhanced_figure(draw_panel3, "quasi_newton_3.pdf")

print("Summary:")
print("x_k =", x_k)
print("g_k =", g_k)
print("B_k =\n", B_k)
print("x_{k+1} =", x_kp1)
print("s_k =", s_k)
print("y_k =", y_k)
print("B_{k+1} =\n", B_kp1)
