#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:58:07 2025

@author: jovanovic with the help of Microsoft Copilot and chatGPT
"""
import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel, QLineEdit, QSlider, QRadioButton, QDialog, QScrollArea, QHBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D
import gmsh
import meshio

# Constants
WINDOW_TITLE = 'photosmehser'
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
CANNY_THRESHOLD_MIN = 0
CANNY_THRESHOLD_MAX = 255
HOUGH_THRESHOLD_MIN = 1
HOUGH_THRESHOLD_MAX = 500
HOUGH_MIN_LENGTH_MAX = 500
HOUGH_MAX_GAP_MAX = 50

class MultiInputDialog(QDialog):
    def __init__(self, data_array):
        super().__init__()
        self.setWindowTitle("Checking corrdinate values")
        self.setGeometry(250, 250, 700, 500)

        # Save the input array
        self.data_array = data_array.astype('str')
        self.updated_values = []
        self.arr_shape = data_array.shape

        print(self.data_array)

        # Create the main layout
        layout = QVBoxLayout()

        # Scrollable area (in case of large arrays)
        scroll_area = QScrollArea(self)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        print(self.data_array.shape)

        # Dynamically create textboxes for each element in the array
        self.textboxes = []
        for value in self.data_array:
            label1 = QLabel(f"Edit x and y values for point: {value}")
            textbox1 = QLineEdit(self)
            textbox2 = QLineEdit(self)
            textbox1.setText(str(value[0]))
            textbox2.setText(str(value[1]))
            self.textboxes.append(textbox1)
            self.textboxes.append(textbox2)
            scroll_layout.addWidget(label1)
            scroll_layout.addWidget(textbox1)
            scroll_layout.addWidget(textbox2)

        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK", self)
        cancel_button = QPushButton("Cancel", self)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        ok_button.clicked.connect(self.accept_dialog)
        cancel_button.clicked.connect(self.reject)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def accept_dialog(self):
        # Collect updated values from textboxes
        self.updated_values = np.reshape(np.asarray([textbox.text() for textbox in self.textboxes]).astype(float), self.arr_shape)
        self.accept()

    def get_updated_values(self):
        # Return the updated values
        return self.updated_values

class SaveFileDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Save File")
        self.setGeometry(300, 300, 400, 100)

        layout = QVBoxLayout()

        # Label to prompt the user
        label = QLabel("Mesh file name:")
        layout.addWidget(label)

        # Textbox for entering the filename
        self.textbox = QLineEdit(self)
        self.textbox.setPlaceholderText("Enter filename without extension!")
        layout.addWidget(self.textbox)

        # OK and Cancel buttons
        self.ok_button = QPushButton("Save", self)
        self.cancel_button = QPushButton("Cancel", self)
        layout.addWidget(self.ok_button)
        layout.addWidget(self.cancel_button)

        # Connect buttons to actions
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        self.setLayout(layout)

    def get_filename(self):
        # Return the entered filename
        return self.textbox.text()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = WINDOW_TITLE
        self.left = 100
        self.top = 100
        self.width = WINDOW_WIDTH
        self.height = WINDOW_HEIGHT
        self.img_rgb = None
        self.img_gray = None
        self.line_coords = []
        self.intersection_points = []
        self.selected_points = []
        self.selected_point_coords = []
        self.conversion_factor = None
        self.length = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.create_widgets()
        self.connect_signals()

    def create_widgets(self):
        self.load_button = QPushButton('Choose File', self)
        self.layout.addWidget(self.load_button)
        self.canvas = FigureCanvas(Figure())
        self.layout.addWidget(self.canvas)

        self.create_intensity_check()
        self.create_canny_sliders()
        self.create_hough_sliders()

        self.close_button = QPushButton('Close', self)
        self.layout.addWidget(self.close_button)

    def create_intensity_check(self):
        self.intensity_check_label = QLabel("Do you want to check if there are parts of the mesh to be refined?", self)
        self.layout.addWidget(self.intensity_check_label)
        self.intensity_yes = QRadioButton("Yes", self)
        self.intensity_no = QRadioButton("No", self)
        self.intensity_no.setChecked(True)  # Default to No
        self.layout.addWidget(self.intensity_yes)
        self.layout.addWidget(self.intensity_no)

    def create_canny_sliders(self):
        self.canny_threshold1_slider = self.create_slider("Canny Threshold1:", CANNY_THRESHOLD_MIN, CANNY_THRESHOLD_MAX, 1)
        self.canny_threshold2_slider = self.create_slider("Canny Threshold2:", CANNY_THRESHOLD_MIN, CANNY_THRESHOLD_MAX, 1)

    def create_hough_sliders(self):
        self.hough_threshold_slider = self.create_slider("Hough Threshold:", HOUGH_THRESHOLD_MIN, HOUGH_THRESHOLD_MAX, HOUGH_THRESHOLD_MAX)
        self.hough_min_length_slider = self.create_slider("Min Line Length:", 0, HOUGH_MIN_LENGTH_MAX, 0)
        self.hough_max_gap_slider = self.create_slider("Max Line Gap:", 0, HOUGH_MAX_GAP_MAX, 50)

    def create_slider(self, label, min_value, max_value, default_value):
        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(min_value, max_value)
        slider.setValue(default_value)
        self.layout.addWidget(QLabel(label))
        self.layout.addWidget(slider)
        return slider

    def connect_signals(self):
        self.load_button.clicked.connect(self.load_image)
        self.close_button.clicked.connect(self.close)
        self.canny_threshold1_slider.valueChanged.connect(self.redraw_lines)
        self.canny_threshold2_slider.valueChanged.connect(self.redraw_lines)
        self.hough_threshold_slider.valueChanged.connect(self.redraw_lines)
        self.hough_min_length_slider.valueChanged.connect(self.redraw_lines)
        self.hough_max_gap_slider.valueChanged.connect(self.redraw_lines)

    def load_image(self):
        self.reset_state()
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "All Files (*);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)", options=options)
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                print("Error: unable to load image.")
                return
            self.img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.redraw_lines()

    def reset_state(self):
        self.line_coords.clear()
        self.intersection_points.clear()
        self.selected_points.clear()

    def redraw_lines(self):
        if self.img_gray is None:
            return
        self.detect_lines(self.img_gray)

    def detect_lines(self, img):
        threshold1 = self.canny_threshold1_slider.value()
        threshold2 = self.canny_threshold2_slider.value()
        hough_threshold = self.hough_threshold_slider.value()
        min_line_length = self.hough_min_length_slider.value()
        max_line_gap = self.hough_max_gap_slider.value()

        edges = cv2.Canny(img, threshold1, threshold2, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

        self.canvas.figure.clf()
        ax = self.canvas.figure.add_subplot(111)
        ax.imshow(self.img_rgb)
        ax.axis('off')

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_patch = Line2D([x1, x2], [y1, y2], linewidth=2, color='lime')
                ax.add_line(line_patch)

        self.canvas.mpl_connect('button_press_event', self.on_pick_line)
        self.canvas.draw()

        if not hasattr(self, 'done_button'):
            self.done_button = QPushButton('Done', self)
            self.done_button.clicked.connect(self.extrapolate_lines)
            self.layout.addWidget(self.done_button)

    def on_pick_line(self, event):
        for artist in self.canvas.figure.get_axes()[0].lines:
            if isinstance(artist, Line2D):
                contains, _ = artist.contains(event)
                if contains:
                    artist.set_color('red')
                    self.line_coords.append([artist.get_xdata(), artist.get_ydata()])
                    self.canvas.draw()
                    break

    def extrapolate_lines(self):
        self.layout.removeWidget(self.done_button)
        self.done_button.deleteLater()
        self.canvas.figure.clf()
        ax = self.canvas.figure.add_subplot(111)
        ax.imshow(self.img_rgb)
        ax.axis('off')

        lines = np.array([[line[0][0], line[1][0], line[0][1], line[1][1]] for line in self.line_coords])

        size = self.img_rgb.shape[:2]
        self.intersection_points = self.calculate_intersections(lines, size)

        for idx, point in enumerate(self.intersection_points):
            ax.plot(point[0], point[1], 'bo')
            ax.text(point[0], point[1], f'{idx}', fontsize=12, color='blue')

        self.canvas.mpl_connect('button_press_event', self.on_pick_point)
        self.canvas.draw()

        if not hasattr(self, 'select_points_button'):
            self.select_points_button = QPushButton('Select Points', self)
            self.select_points_button.clicked.connect(self.ask_for_length)
            self.layout.addWidget(self.select_points_button)

    def calculate_intersections(self, lines, size):
        intersection_points = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                p1, p2 = lines[i][:2], lines[i][2:]
                p3, p4 = lines[j][:2], lines[j][2:]
                ex_p1, ex_p2 = self.extrapolate_line(*p1, *p2, size)
                ex_p3, ex_p4 = self.extrapolate_line(*p3, *p4, size)
                intersection = self.find_intersection(ex_p1, ex_p2, ex_p3, ex_p4)
                if intersection is not None:
                    intersection_points.append(intersection)
        return intersection_points

    def extrapolate_line(self, x1, y1, x2, y2, size):
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            x_left = 0
            y_left = m * x_left + b
            x_right = size[1]
            y_right = m * x_right + b
            return [(x_left, y_left), (x_right, y_right)]
        else:
            return [(x1, 0), (x1, size[0])]

    def find_intersection(self, p1, p2, p3, p4):
        s1 = np.array(p2) - np.array(p1)
        s2 = np.array(p4) - np.array(p3)
        denom = -s2[0] * s1[1] + s1[0] * s2[1]
        if denom == 0:
            return None
        s = (-s1[1] * (p1[0] - p3[0]) + s1[0] * (p1[1] - p3[1])) / denom
        t = (s2[0] * (p1[1] - p3[1]) - s2[1] * (p1[0] - p3[0])) / denom
        if 0 <= s <= 1 and 0 <= t <= 1:
            return p1 + (t * s1)
        return None

    def on_pick_point(self, event):
        for artist in self.canvas.figure.get_axes()[0].lines:
            if isinstance(artist, Line2D):
                contains, _ = artist.contains(event)
                if contains:
                    x, y = artist.get_xdata()[0], artist.get_ydata()[0]
                    self.selected_points.append([x, y])
                    artist.set_color('red')
                    self.canvas.draw()
                    break

    def ask_for_length(self):
        if hasattr(self, 'select_points_button'):
            self.layout.removeWidget(self.select_points_button)
            self.select_points_button.deleteLater()

        self.length_label = QLabel("Enter Length:", self)
        self.layout.addWidget(self.length_label)

        self.length_entry = QLineEdit(self)
        self.layout.addWidget(self.length_entry)

        self.confirm_button = QPushButton('Confirm', self)
        self.confirm_button.clicked.connect(self.confirm_length)
        self.layout.addWidget(self.confirm_button)

    def check_coordinate_values(self):
        # Pass the array to the dialog and retrieve updated values
        dialog = MultiInputDialog(self.selected_point_coords)
        if dialog.exec_():  # If the dialog is accepted
            self.selected_point_coords = dialog.get_updated_values()  # Get the updated array
            print("Updated Array:", self.selected_point_coords)  # Example output

    def confirm_length(self):
        self.length = self.length_entry.text()
        print(f"Length entered: {self.length}")
        self.cleanup_length_widgets()

        selected_intersection_points_array = np.array(self.selected_points)
        print("Selected intersection points:")
        print(selected_intersection_points_array)

        self.selected_point_coords = (float(self.length) * selected_intersection_points_array /
                  (selected_intersection_points_array[0, 1] - selected_intersection_points_array[1, 1]))
        self.conversion_factor = (float(self.length) / (selected_intersection_points_array[0, 1] - selected_intersection_points_array[1, 1]))

        self.selected_point_coords[:, 0] -= self.selected_point_coords[0, 0]
        self.selected_point_coords[:, 1] -= self.selected_point_coords[1, 1]

        if self.intensity_yes.isChecked():
            self.check_intensity()
        self.check_coordinate_values()
        self.create_mesh()
#        self.close()

    def cleanup_length_widgets(self):
        self.length_label.deleteLater()
        self.length_entry.deleteLater()
        self.confirm_button.deleteLater()

    def check_intensity(self):
        global zmin, zmax, rmin, rmax
        if self.img_gray is not None and len(self.intersection_points) > 1:
            min_x = int(min(point[0] for point in self.intersection_points))+5
            max_x = int(max(point[0] for point in self.intersection_points))-5
            min_y = int(min(point[1] for point in self.intersection_points))+5
            max_y = int(max(point[1] for point in self.intersection_points))-5

            # Crop the grayscale image to the bounding box defined by intersection points
            cropped_img = np.flip(self.img_gray, axis = 0)
            cropped_img = cropped_img[min_y:max_y, min_x:max_x]
#            plt.contourf(cropped_img)

            # Find the maximum intensity value
            max_value = np.max(cropped_img)
#            threshold_value = max_value * 0.7

            rz = np.argwhere(cropped_img > 0.9*max_value)

            zmax = rz[:,0].max()
            zmin = rz[:,0].min()
            rmax= rz[:,1].max()

            print(max_value)
            print(zmax)
            print(zmin)
            print(rmax)

    def create_mesh(self):
        points = np.asarray(self.selected_point_coords)
#        points[:, 0] -= points[0, 0]
#        points[:, 1] -= points[1, 1]

        gmsh.initialize()
        gmsh.model.add("mesh")
        lc = 1e-3
        pts = []
        for pt in points:
            pts.append(gmsh.model.geo.addPoint(pt[0], pt[1], 0, lc))
            print(str(pt))

        lins = []
        for idx in range(len(pts)):
            lins.append(gmsh.model.geo.addLine(pts[idx - 1], pts[idx]))
        cl = gmsh.model.geo.addCurveLoop(lins, 13)
        ps = gmsh.model.geo.addPlaneSurface([cl])

        if self.intensity_yes.isChecked():
            ref = 5e-2
            gmsh.model.mesh.field.add("Box", 3)
            gmsh.model.mesh.field.setNumber(3, "VIn", ref * lc)
            gmsh.model.mesh.field.setNumber(3, "VOut", lc)
            gmsh.model.mesh.field.setNumber(3, "XMin", 0.0)
            print('xmin = 0')
            gmsh.model.mesh.field.setNumber(3, "XMax", rmax*self.conversion_factor)
            print('xmax = ' + str(rmax*self.conversion_factor))
            gmsh.model.mesh.field.setNumber(3, "YMin", zmin*self.conversion_factor)
            print('ymin = ' + str(zmin*self.conversion_factor))
            gmsh.model.mesh.field.setNumber(3, "YMax", zmax*self.conversion_factor)
            print('ymax = ' + str(zmax*self.conversion_factor))
            gmsh.model.mesh.field.setNumber(3, "Thickness", 1e-4)
            gmsh.model.mesh.field.add("MathEval", 5)
            gmsh.model.mesh.field.setString(5, "F", "300*F3^2 + " + str(ref * lc))
            gmsh.model.mesh.field.add("Min", 7)
            gmsh.model.mesh.field.setNumbers(7, "FieldsList", [3, 5])
            gmsh.model.mesh.field.setAsBackgroundMesh(7)

        dialog = SaveFileDialog()
        if dialog.exec_():  # If the dialog is accepted
            fname = dialog.get_filename()
            print("Sucecss!")  # Example output


        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate()
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(fname + ".msh")
        gmsh.write(fname + ".geo_unrolled")

        gmsh.model.geo.synchronize()
        if '-nopopup' not in sys.argv:
            gmsh.fltk.run()

        gmsh.finalize()
        mesh = meshio.read(fname + ".msh")
        mesh.write(fname + ".vtk")
        mesh.write(fname + ".xml")
        self.close()  # Close the window

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
