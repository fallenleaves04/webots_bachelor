

import numpy as np
import scipy
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

class BSplineTrajectory:
    def __init__(self, control_points, degree=3, n_points=10):
        """
        control_points: lista punktów kontrolnych w formie [(x0, y0), (x1, y1), ...]
        degree: stopień B-Splina (3 to najczęściej używane dla gładkich trajektorii)
        n_points: liczba punktów, które chcemy uzyskać na trajektorii
        """
        self.control_points = np.array(control_points)
        self.degree = degree
        self.n_points = n_points

        # Ustalamy wektory węzłów (knots), które określają, jak "rozciągają się" punkty kontrolne
        self.knots = np.linspace(0, 1, len(self.control_points) - self.degree + 1)
        print(self.knots)
        # Tworzymy obiekt B-Splina na podstawie punktów kontrolnych i wektorów węzłów
        self.bspline = BSpline(self.knots, self.control_points, self.degree)

    def generate_trajectory(self):
        """
        Generowanie trajektorii na podstawie B-Splina
        """
        t = np.linspace(0, 1, self.n_points)  # Parametryzacja czasu [0, 1]
        trajectory = self.bspline(t)  # Generowanie punktów na trajektorii
        return trajectory

    def plot_trajectory(self):
        """
        Wizualizacja wygenerowanej trajektorii
        """
        trajectory = self.generate_trajectory()
        trajectory_x = trajectory[:, 0]
        trajectory_y = trajectory[:, 1]

        plt.figure(figsize=(8, 6))
        plt.plot(trajectory_x, trajectory_y, label='B-Spline Trajectory', color='blue')
        plt.scatter(self.control_points[:, 0], self.control_points[:, 1], color='red', label='Control Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.title('B-Spline Trajectory')
        plt.show()

# Przykładowe punkty kontrolne (możesz je dowolnie zmieniać)
control_points = [[0, 0], [3, 2], [5, 5], [6, 2], [8, 0],[5,2],[6,7],[2,5],[5, 5], [3, 2], [5, 5]]

# Tworzymy obiekt trajektorii B-Splinem
trajectory_planner = BSplineTrajectory(control_points, degree=3, n_points=100)

# Wyświetlamy trajektorię
trajectory_planner.plot_trajectory()