# A Python3 program to find if 2 given line segments intersect or not

def edge_intersect(e1, e2):
    return edge_intersect(e1, e2, BBox(e1))


def edge_intersect(e1, e2, bb1):
    if bb1.overlaps(BBox(e2)):
        p1 = Point(e1[0].coord_x, e1[0].coord_y)
        p2 = Point(e1[1].coord_x, e1[1].coord_y)

        q1 = Point(e2[0].coord_x, e2[0].coord_y)
        q2 = Point(e2[1].coord_x, e2[1].coord_y)
        return intersect(p1, p2, q1, q2)
    return False


def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


class BBox:
    def __init__(self, edge):
        self.x_min = min(edge[0].coord_x, edge[1].coord_x)
        self.x_max = max(edge[0].coord_x, edge[1].coord_x)

        self.y_min = min(edge[0].coord_y, edge[1].coord_y)
        self.y_max = max(edge[0].coord_y, edge[1].coord_y)

    def overlaps(self, other: 'BBox'):
        if self.x_max < other.x_min: return False
        if self.x_min > other.x_max: return False
        if self.y_max < other.y_min: return False
        if self.y_min > other.y_max: return False
        return True


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y