#!/usr/bin/python
from math import *
import sys
import os

Nx=32*12
Ny=32*12
Nz=32*12
FloatPrecision=1

MPIon=0
execfile('compileSource.py')

import lbm
from lbm import G

Nx,Ny,Nz = G.Nx,G.Ny,G.Nz
print "Grid sizes: %d x %d x %d"%(Nx,Ny,Nz)
G.PPhost.setDefault()

G.PPhost.Nt=16;

G.PPhost.dx=2.0;
G.PPhost.dy=2.0;
G.PPhost.dz=2.0;
G.PPhost.dt=1.0;

#SS = G.PP.src
#SS.srcX, SS.srcY, SS.srcZ = center[0],center[1],center[2];
#SS.set(50.0,cL,pi/2,0)

G.PPhost.StepIterPeriod = 1#256#1024#1#*4#16384;
G.PPhost.set_drop_dir(drop_into);

lbm.run(sys.argv)

############################################################


