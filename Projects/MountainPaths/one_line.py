import heapq, math
from PIL import Image
import numpy as np
child = None
node = None

if __name__ == '__main__':
    file_path = "Colorado_480x480.dat"
    start_coord = (0, 0)
    end_coord = (479, 479)


while True:
    if __name__ == "__main__":
        assign = lambda k, v: globals().__setitem__(k, v)
        file = (np.loadtxt(file_path) if ".dat" in file_path else np.array(Image.open(file_path))).astype(np.float32)
        start_node = {"start": 0, "end": math.dist(end_coord, start_coord), "parent": None, "pos": start_coord, "value": file[start_coord], "x": start_coord[0], "y":start_coord[1]}
        open_list, closed_list = [start_node], set()
        home_distances = dict()
        create_node = lambda p, n: {"start": p["start"] + abs(file[n] - p["value"]), "end": math.dist(end_coord, n), "parent": p, "pos": n, "value": file[n], "x": n[0], "y":n[1]}
        __name__ = file, open_list, closed_list, home_distances, create_node, assign
    else:
        file, open_list, closed_list, home_distances, create_node, assign = __name__
        assign("node", heapq.heappop(open_list))
        closed_list.add(node["pos"])
        # draw node if doing that
        if not node["end"]:
            print("done")
        for neighbor in [(node['x'] + dx, node['y'] + dy) for dx in range(-1, 2) for dy in range(-1, 2)]:
            assign("child", create_node(node, neighbor))
            home_distances.get(child["pos"], child["start"]+1) > child["start"]  \
                and (home_distances.__setitem__(child["pos"], child) or True) \
                    and child["pos"] not in closed_list \
                    and heapq.heappush(open_list, child)

