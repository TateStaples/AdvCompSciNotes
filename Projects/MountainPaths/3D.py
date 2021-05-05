import matplotlib.pyplot as plt
from Mountain import Pathfinder
import numpy as np


class BetterVisual(Pathfinder):
    frequency = None

    def draw(self):
        pass

    def h(self, x, y):
        return self.map[(int(x), int(y))]# got to scale so height at same scale as dx

    def __call__(self, *args, **kwargs):
        # route = self.path()
        # print("done pathing")
        # data = np.array([(*node.pos, self.map[node.pos]) for node in route])
        # np.save("path", np.array(data))
        data = np.load("path.npz.npy")
        self.draw_3d_map(data)

    def draw_3d_map(self, route=None):
        h, w = self.map.shape
        x = np.linspace(0, w, w)
        y = np.linspace(0, h, h)
        X, Y = np.meshgrid(x, y)
        Z = self.map
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.axis('off')
        ax.set_zlim(0, self.map.max() * 3)
        ax.plot_surface(X, Y, Z, rstride=50, cstride=50,
                        cmap='viridis', edgecolor='none')
        # ax.cla()
        if route is not None:
            plt.plot(route[:, 0], route[:, 1], route[:, 2], color="red")
        ax.set_xlabel('longitude')
        ax.set_ylabel('lattitude')
        ax.set_zlabel('height')
        plt.show()


if __name__ == '__main__':
    b = BetterVisual(240, "Colorado_480x480.dat")
    # b.draw_3d_map()
    b()
