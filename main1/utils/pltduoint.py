#-- plot interposer layout for duo chiplet
import os
import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

#-- interposer size
width = 3e-2
height = 3e-2

#-- unit cell size
uwidth = 3e-3
uheight = 3e-3


#-- horizontal lines
N_wire_x = math.floor(height/(uheight))
#-- vertical lines
N_wire_y = math.floor(width/(uwidth))

fig, ax = plt.subplots()

#-- interposer
ax.plot([0,width], [0,height], color='black', alpha=0.0)

#-- horizontal lines
i = 0
while i <= N_wire_x:
    x = [0, width]
    y = [i*height/N_wire_x, i*height/N_wire_x]
    plt.plot(x, y, 'y-')
    i += 1

#-- vertical lines
i = 0
while i <= N_wire_y:
    x = [i*width/N_wire_y, i*width/N_wire_y]
    y = [0, height]
    #plt.plot(x, y, 'y-.', linestyle='--')
    plt.plot(x, y, 'y-')
    i += 1

#-- add chiplet
rx = 10e-3
ry = 10e-3
cwidth = 12e-3
cheight = 12e-3
cx = rx + 0.5*cwidth
cy = ry + 0.5*cheight

ax.add_patch(Rectangle((rx, ry), cwidth, cheight, color="red", alpha=0.5))
ax.annotate("duo", (cx, cy), color='black', weight='bold', fontsize=10, ha='center', va='center')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')


#-- interposer tsv
tsvxno = 20
tsvyno = 20
i = 0
while i < tsvxno:
    x = (i+1)*width/(tsvxno+1)
    j = 0
    while j < tsvyno:
        y = (j+1)*height/(tsvyno+1)
        if (i+j) % 2 == 0:
            plt.plot(x, y, 'o', color='blue', alpha=0.6)
        else:
            plt.plot(x, y, 'o', color='red', alpha=0.6)
        j += 1
    i += 1

plt.show()
