import numpy as np
import matplotlib.pyplot as plt
import itertools
from PIL import Image
import io
from tqdm import tqdm


class RoundGraph:
    def __init__(self, points):
        self.points = points

        self.axis = points.shape[1]
        angle = 180 / self.axis
        angle_by_axis = list(range(0, 180, int(angle)))
        self.radians_by_axis = np.radians(angle_by_axis)

        self.frames = []

        fig, self.ax = plt.subplots()

        # Constant window proportions
        plt.gca().set_aspect('equal', adjustable='box')

    def _value_position_on_axis(self, value, angle_radians):
        x_end = (value / 2) * np.cos(angle_radians)
        y_end = (value / 2) * np.sin(angle_radians)
        return [x_end, y_end]

    def _values_positions_on_axis(self, input_array):
        return np.array([self._value_position_on_axis(input_array[i], self.radians_by_axis[i])
                         for i in range(len(self.radians_by_axis))])

    def _points_on_axis_to_graph(self, point):
        return [point[:, 0].sum() / len(point), point[:, 1].sum() / len(point)]

    def _point_to_graph(self, point):
        return self._points_on_axis_to_graph(self._values_positions_on_axis(point))

    def draw_point(self, point):
        x, y = self._point_to_graph(point)
        self.ax.scatter(x, y, color='blue', marker='o')

    def draw_points(self, points):
        for p in points:
            self.draw_point(p)

    def draw_lines(self, line_points):
        for p in line_points:
            x1, y1 = self._point_to_graph(p[0])
            x2, y2 = self._point_to_graph(p[1])
            plt.plot([x1, x2], [y1, y2])

    def draw_axis(self, length=1, thickness=1):
        axis_lines_points = self._values_positions_on_axis(np.full(self.axis, length))
        axis_names = 'xyzwvu'

        for i, p in enumerate(axis_lines_points):
            plt.text(p[0], p[1], axis_names[i], fontsize=12, color='red')
            plt.plot([-p[0], p[0]], [-p[1], p[1]], color='black', linewidth=thickness)

        plt.axis('off')

    def animation(self, steps=180):
        for i in tqdm(range(steps)):
            angle = i * 360 / steps

            rotation_matrix = np.eye(self.axis)
            rotation_matrix[:2, :2] = [[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                       [np.sin(np.radians(angle)), np.cos(np.radians(angle))]]
            rotated_points = np.dot(self.points, rotation_matrix)

            # Lines
            def distance(point1, point2):
                return np.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

            lines_points = []
            for pair in itertools.combinations(rotated_points, 2):
                if np.isclose(distance(pair[0], pair[1]), 2.0):
                    lines_points.append(list(pair))

            self.ax.cla()
            self.draw_axis()
            self.draw_points(rotated_points)
            self.draw_lines(lines_points)

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')

            self.frames.append(Image.open(img_buf))
            plt.pause(0.000001)

        return self.frames

    def save(self, path):
        self.frames[0].save(path, save_all=True, append_images=self.frames[1:], duration=20, loop=0)


def main():
    square = np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ])

    cube = np.array([[-1, -1, -1],
                     [1, -1, -1],
                     [1, 1, -1],
                     [-1, 1, -1],
                     [-1, -1, 1],
                     [1, -1, 1],
                     [1, 1, 1],
                     [-1, 1, 1]])

    tesseract = np.array([
        [-1, -1, -1, -1],
        [1, -1, -1, -1],
        [1, 1, -1, -1],
        [-1, 1, -1, -1],
        [-1, -1, 1, -1],
        [1, -1, 1, -1],
        [1, 1, 1, -1],
        [-1, 1, 1, -1],
        [-1, -1, -1, 1],
        [1, -1, -1, 1],
        [1, 1, -1, 1],
        [-1, 1, -1, 1],
        [-1, -1, 1, 1],
        [1, -1, 1, 1],
        [1, 1, 1, 1],
        [-1, 1, 1, 1]
    ])

    graph = RoundGraph(tesseract)
    graph.animation()
    graph.save('result/tesseract.gif')


if __name__ == "__main__":
    main()
