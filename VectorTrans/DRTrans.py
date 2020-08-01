
class DRTrans:
    def __init__(self):
        self.X = None
        self.Y = None
        self.label = None
        self.derivative = None
        self.point_error = None
        self.y_add_list = []
        self.y_sub_list = []
        self.k = 0
        self.eigen_number = 2
        # self.error = None
        self.linearity = None

    def transform(self, nbrs_k, MAX_EIGEN_COUNT=5, yita=0.1):
        return self.Y, self.y_add_list, self.y_sub_list


