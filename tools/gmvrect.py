#!/usr/bin/env python

#
# gmvrect.py
# Writes a rectangular gmv mesh with user-specified dimensions
#

import sys

def usage():
    print "Usage:  gmvrect.py NZX [NZY [LENX [LENY]]]"
    print "where nzx, nzy   = number of zones in x, y directions"
    print "                   (no default for nzx; default nzy = nzx)"
    print "      lenx, leny = total length in x, y directions"
    print "                   (default for both = 1.0)"
    sys.exit(0)

nargs = len(sys.argv)
nzx = 0
nzy = 0
lenx = 1.0
leny = 1.0
if nargs < 2 or nargs > 5:  usage()
try:
    nzx = int(sys.argv[1])
    nzy = nzx
    if nargs > 2: nzy = int(sys.argv[2])
    if nargs > 3: lenx = float(sys.argv[3])
    if nargs > 4: leny = float(sys.argv[4])
except:
    usage()
if nzx <= 0 or nzy <= 0 or lenx <= 0. or leny <= 0.:  usage()
    
nz = nzx * nzy
npx = nzx + 1
npy = nzy + 1
np = npx * npy

filename = "rect%dx%d.gmv" % (nzx, nzy)
file = open(filename, "w")

# write header
file.write("gmvinput ascii\n")

# write node header
file.write("nodes  %9d\n" % np)

# write node x coords
for i in range(np):
    ix = i % npx
    if i % 10 == 0:  s = "  "
    s += "%16.8E" % (lenx * float(ix) / nzx)
    if (i % 10 == 9) or (i == np - 1):  file.write("%s\n" % s)

# write node y coords
for i in range(np):
    iy = i / npx
    if i % 10 == 0:  s = "  "
    s += "%16.8E" % (leny * float(iy) / nzy)
    if (i % 10 == 9) or (i == np - 1):  file.write("%s\n" % s)

# write node z coords (always 0 in 2D)
for i in range(np):
    if i % 10 == 0:  s = "  "
    s += "  0.00000000E+00"
    if (i % 10 == 9) or (i == np - 1):  file.write("%s\n" % s)

# write cell header
file.write("cells  %9d\n" % nz)

# write cells
for i in range(nz):
    ix = i % nzx
    iy = i / nzx
    file.write("  general          1\n")
    file.write("             4\n")
    p0 = iy * npx + ix + 1
    p1 = p0 + 1
    p2 = p1 + npx
    p3 = p2 - 1
    file.write("     %9d %9d %9d %9d\n" % (p0, p1, p2, p3))

# write end-of-file marker
file.write("endgmv\n")

print "Wrote %s" % filename
