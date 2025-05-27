import sys
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *

ifile = "image.png"
ofile = "model.bin"

image = plt.imread(ifile)
nz, nx, rgb = image.shape
model = np.ones((nz, nx))

class BoxDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Build your model layer by layer")
        self.setGeometry(650, 400, 400, 100)

        font = QFont("Arial", 15)

        label = QLabel("Set a property for the current layer:")
        label.setFont(font)

        self.input_property = QLineEdit(self)
        self.input_property.setPlaceholderText("Enter a model property")
        self.input_property.setFont(font)

        self.button = QPushButton("OK", self)
        self.button.setFont(font)
        self.button.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.input_property)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def get_property(self):
        return float(self.input_property.text())

def on_click(event):
    z = int(event.ydata)
    x = int(event.xdata)
    v = image[z, x]

    R_mask = image[:, :, 0] == v[0]
    G_mask = image[:, :, 1] == v[1]
    B_mask = image[:, :, 2] == v[2]
    mask = R_mask & G_mask & B_mask

    app = QApplication.instance() or QApplication(sys.argv)
    
    dialog = BoxDialog()

    if dialog.exec_() == QDialog.Accepted:
        model[mask] = dialog.get_property()

        ax[0].imshow(image, aspect = "auto", cmap = "jet")
        ax[1].imshow(model, aspect = "auto", cmap = "jet")
        fig.canvas.draw()

fig, ax = plt.subplots(ncols = 2, figsize = (15, 4))

ax[0].imshow(image, aspect = "auto", cmap = "jet")
ax[1].imshow(model, aspect = "auto", cmap = "jet")

fig.canvas.mpl_connect("button_press_event", on_click)
fig.tight_layout()
plt.show()

z, x = np.where(model == 1.0)

for k in range(len(z)):
    zs = slice(z[k] - 2, z[k] + 3)
    neighbours = model[zs, x[k]]
    data = neighbours[neighbours > 1.0]
    model[z[k], x[k]] = np.sum(data) / np.size(data)

fig, ax = plt.subplots(figsize = (8, 4))
im = ax.imshow(model, aspect = "auto", cmap = "jet")
cbar = plt.colorbar(im, ax = ax)
fig.tight_layout()
plt.show()

model.flatten("F").astype(np.float32, order = "F").tofile(ofile)