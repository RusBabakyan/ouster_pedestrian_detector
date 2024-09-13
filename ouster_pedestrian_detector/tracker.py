import numpy as np

from collections import defaultdict


class TrackedObject:
    def __init__(self, cart_position, id, lost) -> None:
        self.cart_position = np.array(cart_position)
        self.id = np.array(id).astype(np.int32)
        self.lost = np.array(lost).astype(bool)
                                

class Point:
    def __init__(self, position: np.array, lost_time: int = 20) -> None:
        self.lost_time = lost_time
        self.position = position
        self.velocity = np.array((0,0))
        self.lost_counter = 0
        self.lost = False

    def update(self, position):
        self.velocity = position - self.position
        self.position = position
        self.lost = False
        self.lost_counter = 0
        pass

    def predict(self):
        if self.increase_lost_counter():
            return True
        else:
            self.lost = True
            self.position += self.velocity * (self.lost_time - self.lost_counter + 1) / self.lost_time
            return False


    def increase_lost_counter(self):
        self.lost_counter += 1
        if self.lost_counter > self.lost_time:
            return True


class Tracker:
    def __init__(self, distance_threshold=0.3, lost_time = 10):
        self.points = defaultdict(lambda: Point())
        self.lost_time = lost_time
        self.index_counter = 0
        self.distance_threshold = distance_threshold

    def new_id(self) -> int:
        self.index_counter += 1
        return self.index_counter

    def track(self, people):
        if self.points:
            if people:
                self.associate(people)
            else:
                self.predict(self.points.keys())

        elif people:
            for position in people.cart_position:
                self.points[self.new_id()] = Point(position, self.lost_time)

        return self.output()
    
    def delete_lost(self, ids):
        if ids:
            for del_id in ids:
                del self.points[del_id]

    def associate(self, people):
        keys   = list(self.points.keys())
        # points_old = np.array(list(self.points.items()))
        points_old = self.to_numpy()
        points_new = people.cart_position
        associations, old_losted_ids, new_appeared_ids = self.dist_matrix(points_old, points_new)
        for old_id, new_id in associations:
            self.points[keys[old_id]].update(points_new[new_id, :])
        
        # self.predict(keys[old_losted_ids])
        self.predict([keys[id] for id in old_losted_ids])

        for new_appeared_id in new_appeared_ids:
            self.points[self.new_id()] = Point(points_new[new_appeared_id, :], self.lost_time)

    def predict(self, ids):
        if not ids:
            return
        del_list = []
        for id in ids:
            if self.points[id].predict():
                del_list.append(id)
        if del_list:
            for id in del_list:
                del self.points[id]

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
    
    def to_numpy(self):
        return np.array(list(map(lambda point: point.position, self.points.values())))
    
    def output(self):
        lost = []
        positions = []
        ids = []
        for id, point in self.points.items():
            ids.append(id)
            positions.append(point.position)
            lost.append(point.lost)

        return TrackedObject(positions, ids, lost)

            
