import json
import os
import numpy as np
from VectorTrans.DRTrans import DRTrans
from VectorTrans import Metric
from Tools import GrahamScan
from Tools import b_spline


class JsonFile:
    def __init__(self, path=""):
        if not os.path.exists(path):
            print("Error: the path doesn't exist!")
            return
        self.path = path

    def create_file(self, dr_trans=DRTrans()):
        print("Create json files...")
        (n, d) = dr_trans.X.shape

        angles = Metric.angle_v1_v2(dr_trans.y_add_list[0], dr_trans.y_add_list[1], dr_trans.Y)
        trust = Metric.trustworthniess(dr_trans.X, dr_trans.Y, dr_trans.k)
        linearity = dr_trans.linearity
        linearity_equal = Metric.linearity_equalize(linearity)

        result_list = []
        json_file = open(self.path + "result.json", 'w', encoding='utf-8')
        json_file.write("[")
        for i in range(0, n):
            item = {}
            item["x"] = dr_trans.Y[i, 0]
            item["y"] = dr_trans.Y[i, 1]
            item["class"] = int(dr_trans.label[i])
            item["dNum"] = d
            item["hdata"] = dr_trans.X[i, :].tolist()
            item["k"] = dr_trans.k

            # convex hull and B-spline
            y_points = []
            y_points.append(dr_trans.Y[i, :])
            for temp_y in dr_trans.y_add_list:
                y_points.append(temp_y[i, :])
            for temp_y in dr_trans.y_sub_list:
                y_points.append(temp_y[i, :])
            convex_hull = GrahamScan.graham_scan(y_points)
            spline = b_spline.bspline(convex_hull, n=100, degree=3, periodic=True)
            spline_aspect = b_spline.spline_aspect(spline)
            item["pointsNum"] = len(spline)
            item["polygon"] = spline.tolist()
            item["polygonSize"] = Metric.centrosymmetry_area(np.array(spline), dr_trans.Y[i, :])
            item["starPolygonSize"] = item["polygonSize"]  # no use
            item["linearProject"] = spline_aspect
            item["linearChange"] = Metric.linearity_change(spline_aspect, linearity[i])

            item["angles"] = angles[i, 0]
            item["angle12Sin"] = np.sin(angles[i, 0]/180.0*np.pi)
            item["knn_keep"] = trust[i, 0]
            item["eigenNumber"] = dr_trans.eigen_number
            item["proportion"] = 1.0  # no use
            item["angleAddSub"] = 180.0  # no use
            item["angleAddSub_basedcos"] = 1.0  # no use
            item["angleAddSub_cosweighted"] = 1.0  # no use
            item["test_attr"] = 1.0  # no use
            item["int_attr"] = 0  # no use
            item["notes"] = 0  # no use
            item["stress"] = []  # no use
            item["error"] = dr_trans.point_error[i, 0]

            y_points1 = []
            y_points2 = []
            for temp_y in dr_trans.y_add_list:
                y_points1.append((temp_y[i, :]-dr_trans.Y[i, :]).tolist())
            item["y_add_points"] = y_points1
            for temp_y in dr_trans.y_sub_list:
                y_points2.append((temp_y[i, :]-dr_trans.Y[i, :]).tolist())
            item["y_sub_points"] = y_points2

            item["linearity"] = linearity[i, 0]
            item["linearityEqualized"] = linearity_equal[i, 0]

            # temp_json = json.dumps(item)
            json_file.write(str(item).replace('\'', '\"'))
            if i < n-1:
                json_file.write(",\n")
        json_file.write("]")
        json_file.close()

        min_x = np.min(dr_trans.Y[:, 0])
        max_x = np.max(dr_trans.Y[:, 0])
        min_y = np.min(dr_trans.Y[:, 1])
        max_y = np.max(dr_trans.Y[:, 1])

        x_low = min_x - 0.1 * (max_x - min_x)
        x_up = max_x + 0.1 * (max_x - min_x)
        y_low = min_y - 0.1 * (max_y - min_y)
        y_up = max_y + 0.1 * (max_y - min_y)

        scale_file = open(self.path+"scale.json", 'w')
        scale_file.write("{" + "\"min_x\":" + str(x_low) + ", \"max_x\":" + str(x_up) + ", \"min_y\":" + str(
            y_low) + ", \"max_y\":" + str(y_up) + "}")
        scale_file.close()


def test():
    data1 = {
        "x": 0.15,
        "hdata": [1, 2, 3, 4],
        "lists": [[1, 2, 3, 4], [6, 7, 8, 9]],
        "stress": []
    }
    data2 = {
        "x": 0.15,
        "hdata": [1, 2, 3, 4],
        "lists": [[1, 2, 3, 4], [6, 7, 8, 9]],
        "stress": []
    }
    lists = [data1, data2]

    json_file = open("F:\\test.json", 'w', encoding='utf-8')
    json_file.write("[")
    for i in range(0, 2):
        temp_json = json.dumps(lists[i])
        json_file.write(str(temp_json).replace('\'', '\"'))
        if i<1:
            json_file.write(",\n")

    json_file.write("]")
    json_file.close()


if __name__ == '__main__':
    test()


