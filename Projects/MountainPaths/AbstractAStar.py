import math
import heapq


class AStar:
    active = None
    start = None
    end = None

    open_list = list()
    closed_list = set()

    def __init__(self, start, end, map):
        AStar.active = self
        self.start = start
        self.end = end
        self.map = map

    def path(self):
        start_node = Node(None, self.start)
        open_list, closed_list = self.open_list, self.closed_list
        home_distances = {start_node: 0}
        if start_node not in open_list: open_list.append(start_node)

        while open_list:
            node = heapq.heappop(open_list)
            closed_list.add(node)
            self.on_close_node(node)
            if self.over(node): return self._trace(node)

            for neighbor, dis in self.neighbors(node.pos):
                child = Node(node, neighbor, dis)

                if child in home_distances and home_distances[child] > child.dis_to_start:
                    home_distances[child] = child.dis_to_start
                    heapq.heappush(open_list, child)
                    self.on_open_node(child)
                elif child not in closed_list:
                    home_distances[child] = child.dis_to_start
                    heapq.heappush(open_list, child)
                    self.on_open_node(child)
                else:
                    del child

    def on_close_node(self, node):
        pass

    def on_open_node(self, node):
        pass

    @staticmethod
    def _trace(node):
        path = list()
        while node.parent is not None:
            path.append(node)
            node = node.parent
        return path[::-1]

    @staticmethod
    def get_index(list, value):
        for count, node in enumerate(list):
            if node.value >= value:
                return count
        return len(list)

    def neighbors(self, pos):
        r, c = pos
        for dc in range(-1, 2):
            for dr in range(-1, 2):
                if dc or dr:
                    new_pos = r + dr, c + dc
                    if self.valid(new_pos):
                        yield new_pos, self.score(pos, new_pos)

    def valid(self, pos):
        h, w = self.map.shape
        return 0 <= pos[0] < h and 0 <= pos[1] < w

    def score(self, pos1, pos2):
        return distance(pos1, pos2)

    def over(self, node):
        # print(node, "----", self.distance(node.pos), node.pos, self.end)
        return not self.distance(node.pos)

    def distance(self, pos):
        return distance(self.end, pos)

    def draw(self):
        pass


class Node:
    start_weight = 1
    end_weight = 1

    def __init__(self, parent, pos, dis=1):
        self.parent = parent
        self.pos = pos
        self.children = list()
        self.dis_to_start = 0 if self.parent is None else self.parent.dis_to_start + dis
        self.dis_to_end = AStar.active.distance(self.pos)
        self.value = self.dis_to_start + self.dis_to_end

    def __call__(self, *args, **kwargs):
        return self.value
    @property
    def x(self):
        return self.pos[1]

    @property
    def y(self):
        return self.pos[0]

    def __eq__(self, other):
        return self.pos == other.pos

    def __repr__(self):
        return f"AStar Node @ {self.pos} with value of {self.value}"

    def __iter__(self):
        return self.pos.__iter__()

    def __hash__(self):
        return hash(self.pos)

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value


def distance(pos1, pos2):
    return math.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pos1, pos2)))


class Test:
    def __init__(self, **kwargs):
        self.stuff = dict()
        for key in kwargs:
            self.__setattr__(key, kwargs[key])
            self.stuff[key] = kwargs[key]
    def __repr__(self):
        return str(self.stuff)

    def __hash__(self):
        print("test")
        return 1


if __name__ == '__main__':
    print(hash(11.4) - hash(10.3))
    quit()
    AStar((1, 1), (4, 4), [])
    test = set()
    n1 = Node(None, (1, 2))
    n2 = Node(None, (1, 1))
    n3 = Node(None, (1, 3))
    test.add(n1)
    test.add(n2)
    test.add(n3)
    print(test.pop())
    print(test)
