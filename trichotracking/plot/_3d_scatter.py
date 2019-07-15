from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

input = mdfo[mdfo.v_pos_abs<4]
ax.scatter(input.tlol_peaks, input.v_pos_abs, input.tl, marker='o')
ax.set_xlabel('Time')
ax.set_ylabel('V')
ax.set_zlabel('L')
plt.show()
