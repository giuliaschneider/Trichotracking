import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from minboxes import minDistMinBox


def test():
    box1 = np.array([[2943, 2573], [2942, 2566], [3084, 2544], [3085, 2551]])
    box2 = np.array([[3036, 2575], [3034, 2562], [3263, 2531], [3265, 2543]])

    dist = minDistMinBox(box1, box2)

    print("Minimal distance is {}".format(dist))
    print("Distnce = {}".format(la.norm(box1[-1] - [3091, 2554])))
    plt.plot(box1[:,0], box1[:,1])
    plt.plot(box2[:,0], box2[:,1])
    plt.show()



test()