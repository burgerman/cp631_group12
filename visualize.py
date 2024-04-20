from bokeh.plotting import figure, show, output_file
from bokeh.layouts import layout
from bokeh.models import Button, ColumnDataSource
from bokeh.io import curdoc
import numpy as np
import os

# Directory that has files
directory_path = 'output'

# we are loading data from files
data_files = sorted([f for f in os.listdir(directory_path) if f.startswith('forest_')])

# Preparing a source to hold data, which can be updated
source = ColumnDataSource(data={'image': [np.zeros((10, 10))], 'x': [0], 'y': [0], 'dw': [10], 'dh': [10]})

# Setting up  the figure
p = figure(x_range=(0, 10), y_range=(0, 10))
p.image(image='image', x='x', y='y', dw='dw', dh='dh', palette=["white", "green", "red", "black"], source=source)

# Index of currently displayed data
index = [0]

def load_data(i):
    #Loading the data from the ith file and updating the plot 
    data = np.loadtxt(os.path.join(directory_path, data_files[i]))
    source.data = {'image': [data], 'x': [0], 'y': [0], 'dw': [10], 'dh': [10]}

def update_plot(attr, old, new):
    #Upadating the the plot based on the slider or button. """
    load_data(index[0])

def go_next():
    #If you want to go to the next image. """
    if index[0] < len(data_files) - 1:
        index[0] += 1
        update_plot(None, None, None)

def go_prev():
    #If you want to go to the previous image.
    if index[0] > 0:
        index[0] -= 1
        update_plot(None, None, None)

# Navigation Buttons
next_button = Button(label="Next", button_type="success")
prev_button = Button(label="Previous", button_type="warning")
next_button.on_click(go_next)
prev_button.on_click(go_prev)

# Loading the Initial data
load_data(0)

# Arranging the layout
ly = layout([
    [p],
    [prev_button, next_button]
])

# Adding to the current document
curdoc().add_root(ly)
