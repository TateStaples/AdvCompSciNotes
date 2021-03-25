from AbstractAStar import AStar, Node
import numpy as np
import cv2


class Pathfinder(AStar):
    def __init__(self, row, file):
        super(Pathfinder, self).__init__((row, 0), None, np.loadtxt(file).astype(np.float32))
        # Node.end_weight = self.average_change() / 2
        self.img = self.grey_to_img(self.map)
        self.process_img = self.img.copy()
    
    def score(self, pos1, pos2):
        score = abs(self.map[pos1] - self.map[pos2])
        # print(score)
        return score

    def distance(self, pos):
        return self.map.shape[1] - pos[1] - 1

    def __call__(self, *args, **kwargs):
        # starting from specific row
        route1 = self.path()
        self.draw_route(route1, (0, 255, 0))
        cv2.imwrite("result.png", self.img)
        print("done route one", route1)
        cv2.imwrite("process.png", self.process_img)

        self.open_list.clear()
        self.closed_list.clear()
        h, w = self.map.shape
        self.open_list = [Node(None, (row, 0)) for row in range(h)]
        route2 = self.path()
        self.draw_route(route2)

        cv2.imwrite("result.png", self.img)

    def draw_route(self, route, color=(255, 0, 0)):
        for node in route:
            self.img[node.pos, :] = color

    @property
    def max(self):
        return self.map.max()

    @property
    def min(self):
        return self.map.min()

    @staticmethod
    def grey_to_img(map):
        map = map - map.min()
        map = map / map.max() * 255
        return cv2.cvtColor(map, cv2.COLOR_GRAY2RGB)

    def on_close_node(self, node):
        return
        self.process_img[(*node.pos, slice(None))] = (0, 0, 255)
        cv2.imshow("test", self.process_img/255)
        cv2.imwrite("process.png", self.process_img)
        if cv2.waitKey(1):
            pass

    def average_change(self):
        vertical_change = abs(self.map[:-1, :] - self.map[1:, :])
        horizontal_change = abs(self.map[:, :-1] - self.map[:, 1:])
        neg_diagonal_change = abs(self.map[:-1, :-1] - self.map[1:, 1:])
        pos_diagonal_change = abs(self.map[:-1, 1:] - self.map[1:, :-1])
        return sum([change.mean() for change in (vertical_change, horizontal_change, neg_diagonal_change, pos_diagonal_change)]) / 4


def greedy(file_name, pos):
    while eval(pos)[1] < np.loadtxt(file_name).shape[1]-1: yield eval(pos) if pos.replace(pos, str(min([(eval(pos)[0] + dy, eval(pos)[1] + 1) for dy in range(-1, 2)], lambda x: abs(np.loadtxt(file_name)[x]-np.loadtxt(file_name)[eval(pos)])))) else None


if __name__ == '__main__':
    # print(test[:-1])
    p = Pathfinder(240, "Colorado_480x480.dat")
    p()
        