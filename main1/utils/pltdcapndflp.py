#-- plot decap node candidates with floorplan map
import os
import sys
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
#from matplotlib.mlab import griddata

if len(sys.argv) < 3:
    print("Plot decap node with floorplan")
    print("Usage: python3 pltdcapndflp.py <floorplan_file> <decap_file> [width] [height]")
    sys.exit(1)

#-- read floorplan file
#-- format: unitname\tdx\tdy\tx0\ty0
def readCSVfile(file):
    blocks = list(csv.reader(open(file, 'r'), delimiter='\t'))
    return blocks

blocks = readCSVfile(sys.argv[1])

#-- read decap file (location & value)
x = np.genfromtxt(sys.argv[2], usecols=(0))
y = np.genfromtxt(sys.argv[2], usecols=(1))
z = np.genfromtxt(sys.argv[2], usecols=(2))

#-- read decap file (node name)
decaps = readCSVfile(sys.argv[2])

if 5 == len(sys.argv):
    chipwidth = float(sys.argv[3])
    chipheight = float(sys.argv[4])

total_decap = np.sum(z)

#-- print total decap (1/2 since we count both vdd and vss currents)
print("Total decap: %e[F]" % total_decap)

#-- print statistics
print("Min decap: %e[F]" % np.min(z))
print("Avg decap: %e[F]" % np.average(z))
print("Max decap: %e[F]" % np.max(z))


fig, ax = plt.subplots()

#-- plot functional blocks
# build a rectangle for each unit in axes coords
i = 0
for e in blocks:
    #-- skip blank lines
    if len(blocks[i]) == 0:
        i += 1
        continue
    #-- skip comment lines
    if blocks[i][0][0] == '#':
        i += 1
        continue
    name = blocks[i][0]
    left, width = float(blocks[i][3]), float(blocks[i][1])
    bottom, height = float(blocks[i][4]), float(blocks[i][2])
    right = left + width
    top = bottom + height

    ax.add_patch(Rectangle((left, bottom), width, height,
        fill=False, clip_on=False,
        edgecolor='black', facecolor='blue', lw=1))
    ax.text(0.5*(left+right), 0.5*(bottom+top), name,
        fontsize=10, color='black')
    i += 1


#-- set axis range
if 5 == len(sys.argv):
    plt.xlim(0, chipwidth)
    plt.ylim(0, chipheight)
else:
    plt.xlim([0, 1.4*np.max(x)])
    plt.ylim([0, 1.4*np.max(y)])

#-- determine symbol size
slope = (10-100)/(math.log10(10000)-math.log10(100))
legendsize = 100 + (int)(slope*(math.log10(len(z)) - 2))
if (legendsize > 100):
    legendsize = 100
if (legendsize < 10):
    legendsize = 10

legendsize = 10

plt.scatter(x, y, c=z, s=legendsize, marker='s', alpha=1.0, cmap='viridis')
#plt.scatter(x, y, c=z, s=legendsize, marker='H', alpha=1.0, cmap='viridis')
#plt.scatter(x, y, c=z, s=10, marker='H', alpha=1.0, cmap='viridis')
#plt.scatter(x, y, c=z, s=100, marker='H', alpha=1.0, cmap='viridis')
#plt.scatter(x, y, c=z, s=100, marker='H', alpha=1.0, cmap='seismic')


plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

plt.grid(linestyle='--', alpha=0.5)
plt.colorbar()
plt.title("Decap plot (%s)" % sys.argv[2])
plt.xlabel("X [m]")
plt.ylabel("Y [m]")

plt.show()


#-- plot histogram
n_bins = 20

print("No. of decaps: %d" % len(z))

# Creating histogram
#fig, axs = plt.subplots(1, 1, figsize =(10, 7), tight_layout = True)
fig, axs = plt.subplots(1, 1)

axs.hist(z, bins = n_bins)

plt.grid(linestyle='--', alpha=0.5)
plt.title("Decap histogram (%s)" % sys.argv[1])
plt.xlabel("Capacitance [F]")
plt.ylabel("Cap #")

# Show plot
plt.show()

