import numpy as np

# Parameters
c_m = 5
pairs = [(1, 1), (10, 100), (100, 10000), (1000, 1000000)]
eps32 = np.finfo(np.float32).eps
eps64 = np.finfo(np.float64).eps
eps_list = [(eps32, "32-bit"), (eps64, "64-bit")]


def gnorm_bound(c_m, L, meps, f):
    return 0.5 * np.sqrt((L * c_m * meps) * f)


def fmt_sci(x):
    s = f"{x:.2e}"
    mant, exp_str = s.split("e")
    exp = int(exp_str)
    return f"{mant}\\times10^{{{exp:+d}}}"


# Build LaTeX
latex = []
latex.append(r"\begin{table}[t]")
latex.append(r"\centering")
latex.append(
    r"\caption{The approximated lower bounds on the gradient norm $\norm{g_k}$ for which algorithms would be stable.}"
)
latex.append(r"\label{tab:grad-bound}")
latex.append(r"\begin{tabular}{cc}")
subtables = []

for meps, name in eps_list:
    header = f"{name} ($\\meps={fmt_sci(meps)}$)"
    sub = []
    sub.append(r"\begin{tabular}{lcc}")
    sub.append(r"\toprule")
    sub.append(rf"\multicolumn{{3}}{{c}}{{{header}}}\\")
    sub.append(
        rf"$L$ & $f(x_k)$ & $\frac12 \sqrt{{L \cdot {c_m} \cdot \meps f(x_k)}}$ \\"
    )
    sub.append(r"\midrule")
    for L, f in pairs:
        assert np.isclose(L, pow(10, int(np.log10(L))))
        assert np.isclose(f, pow(10, int(np.log10(f))))
        L_str = f"$10^{{{int(np.log10(L))}}}$"
        f_str = f"$10^{{{int(np.log10(f))}}}$"
        val = fmt_sci(gnorm_bound(c_m, L, meps, f))
        sub.append(f"{L_str} & {f_str} & ${val}$ \\\\")
    sub.append(r"\bottomrule")
    sub.append(r"\end{tabular}")
    subtables.append("\n".join(sub))

latex.append(subtables[0] + " & " + subtables[1])
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")

print("\n".join(latex))
