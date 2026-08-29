import matplotlib.pyplot as plt
import numpy as np

from qnlab.experiment.profile import data_profile


def test_data_profile_normalizes_calls_by_dimension_plus_one() -> None:
    _, handles = data_profile(
        np.array([[3.0, np.inf], [6.0, 3.0]]),
        np.array([2, 2]),
        linestyle=["-", "--"],
        colors=["red", "blue"],
        alpha_max=2.0,
    )

    assert handles[0][0].get_xdata().tolist() == [0.0, 1.0, 2.0]
    assert handles[0][0].get_ydata().tolist() == [0.0, 0.5, 1.0]
    assert handles[1][0].get_xdata().tolist() == [0.0, 1.0]
    assert handles[1][0].get_ydata().tolist() == [0.0, 0.5]
    plt.close()
