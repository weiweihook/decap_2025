#-- plot floorplan from a hotspot flp file

import os
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


if len(sys.argv) != 2:
    print("Usage: python3 pltflp <flp_file>")
    sys.exit(1)


#-- read floorplan file
#-- format: unitname\tdx\tdy\tx0\ty0
def readflpfile(file):
    entries = list(csv.reader(open(file, 'r'), delimiter='\t'))
    return entries


entries = readflpfile(sys.argv[1])

#-- find floorplan boundary
max_x = 0
max_y = 0
i = 0
for e in entries:
    #-- skip blank lines
    if len(entries[i]) == 0:
        i += 1
        continue
    #-- skip comment lines
    if entries[i][0][0] == '#':
        i += 1
        continue
    name = entries[i][0]
    left, width = float(entries[i][3]), float(entries[i][1])
    bottom, height = float(entries[i][4]), float(entries[i][2])
    right = left + width
    top = bottom + height
    if max_x < right:
        max_x = right
    if max_y < top:
        max_y = top
    i += 1

#-- define figure and axis
fig, ax = plt.subplots()

#-- draw a transparent diagonal line of the entire floorplan
#-- alpha: transparency: 1=fully visible, 0=invisible
ax.plot([0,max_x], [0,max_y], color='white', alpha=0.0)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')

# build a rectangle for each unit in axes coords
i = 0
for e in entries:
    #-- skip blank lines
    if len(entries[i]) == 0:
        i += 1
        continue
    #-- skip comment lines
    if entries[i][0][0] == '#':
        i += 1
        continue
    name = entries[i][0]
    left, width = float(entries[i][3]), float(entries[i][1])
    bottom, height = float(entries[i][4]), float(entries[i][2])
    right = left + width
    top = bottom + height

    #ax.add_patch(Rectangle((0.0002, 0.0002), 0.0004, 0.0004))
    ax.add_patch(Rectangle((left, bottom), width, height,
        fill=False, clip_on=False,
        edgecolor='blue', facecolor='blue', lw=3))
    ax.text(0.5*(left+right), 0.5*(bottom+top), name,
        fontsize=10, color='red')

    i += 1

plt.show()
