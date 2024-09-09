import numpy as np

from collections import namedtuple, defaultdict

class Point:

    lost_time = 20

    def __init__(self, position: np.array) -> None:
        self.position = position
        self.velocity = np.array((0,0))
        self.lost_counter = 0

    def update(self):
        pass

    def predict(self):
        if self.increase_lost_counter():
            return True
        else:
            pass

    def increase_lost_counter(self):
        self.lost_counter += 1
        if self.lost_counter > self.lost_time:
            return True


class Tracker:
    def __init__(self, distance_threshold=0.1):
        self.points = defaultdict(lambda: Point())
        self.index_counter = 0
        self.distance_threshold = distance_threshold

    def new_id(self) -> int:
        self.index_counter += 1
        return self.index_counter

    def input(self, people):
        if self.points:
            if people:
                self.associate(people)
                pass
            else:
                for id in self.points:
                    self.predict(id)
        else:
            for position in people.cart_position:
                self.points[self.new_id()] = Point(position)

    def associate(self, people):
        keys   = list(self.points.keys())
        points_old = list(self.points.items())
        points_new = people.cart_position
        associations, old_losted_ids, new_appeared_ids = self.dist_matrix(points_old, points_new)
        for old_id, new_id in associations:
            self.points[keys[old_id]].update(points_new[new_id, :])

        for old_losted_id in old_losted_ids:
            self.predict(keys[old_losted_id])

        for new_appeared_id in new_appeared_ids:
            self.points[new_id()] = Point(points_new[new_appeared_id, :])
        pass

    def predict(self, id):
        if self.points[id].predict():
            del self.points.pop(id)

    def dist_matrix(self, L1, L2):
        dist = np.linalg.norm(L1[:, None, :] - L2[None, :, :], axis = -1)
        rows_amount, cols_amount = dist.shape

        associations = []
        old = []
        new = []
        max_value = np.max(dist)*2
        for _ in range(rows_amount):
            min_index = np.argmin(dist)
            row, col = np.unravel_index(min_index, dist.shape)
            if dist[row, col] < self.distance_threshold:
                dist[row, :] = max_value
                dist[:, col] = max_value
                associations.append((row, col))
                old.append(row)
                new.append(col)
            else:
                break
        old_losted   = list(set(range(rows_amount)) - set(old))
        new_appeared = list(set(range(cols_amount)) - set(new))
        return associations, old_losted, new_appeared

            
