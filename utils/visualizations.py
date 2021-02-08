import matplotlib.pyplot as plt
from typing import List, Dict, Union
import numpy as np

def plot_results(x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray], meta_data: Dict) -> None:
    plt.figure()
    plt.plot(x, y)
    plt.title(meta_data["title"])
    plt.xlabel(meta_data["x_label"])
    plt.ylabel(meta_data["y_label"])
    plt.show()
