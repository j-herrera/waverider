# Import modules
import sys
sys.path.append('../')

import numpy as np
from waverider import conical_flow as cone
from scipy.integrate import solve_ivp

import cadquery as cq


# Define constants
Mach = 4
gamma = 1.4
cone_angle = 20 * np.pi / 180
N_stream = 20


# Compute leading edge 
x_apex = 20
x_back = x_apex + 40

shock_angle = cone.shock_angle(Mach, cone_angle, gamma)[0]

z_le = - x_apex * np.tan(shock_angle)

x_le = 0.5 * (1 - np.cos(np.linspace(0, 1, N_stream) * np.pi)) * (x_back - x_apex) + x_apex
y_le = np.zeros(N_stream)

a = x_apex
b = a * np.tan(shock_angle)

y_le = np.sqrt(b**2 * (x_le**2/a**2 - 1))


# Propagate streamlines from the leading edge through the conical flow
[Vr, Vt] = cone.cone_field(Mach, cone_angle, shock_angle, gamma)

def stode(t, x):
    th = np.arctan(x[1] / x[0])
    
    dxdt = np.zeros(2)
    
    dxdt[0] = Vr(th) * np.cos(th) - np.sin(th) * Vt(th)
    dxdt[1] = Vr(th) * np.sin(th) + np.cos(th) * Vt(th)
    
    return dxdt
    
def back(t, y):
    return y[0] - x_back
    
back.terminal = True
    
eta_le = np.sqrt(z_le**2 + y_le**2)
alphas = np.arctan2(y_le, -z_le)

streams = []
for i in range(N_stream):
    sol = solve_ivp(stode, (0, 1000), [x_le[i], eta_le[i]], events=back, max_step=2)
    streams.append(np.vstack([sol.y[0], sol.y[1] * np.sin(alphas[i]), -sol.y[1] * np.cos(alphas[i])]).T)
   
    
## Export waverider CAD model in STEP
# Compute leading edge
le_p  = np.vstack(x[0] for x in streams)
le_n = np.flip(le_p.copy(), 0)
le_n[:, 1] = -le_n[:, 1]
le = np.vstack([le_n[:-1, :], le_p]) # avoid double center point

# Compute trailing edge
te_p  = np.vstack(x[-1] for x in streams)
te_n = np.flip(te_p.copy(), 0)
te_n[:, 1] = -te_n[:, 1]
te = np.vstack([te_n[:-1, :], te_p])

# Add interior points
surface_points = []
for i in range(len(streams)):
    for j in range(1, streams[i].shape[0]-1):
        surface_points.append(tuple(streams[i][j]))

   
for i in range(1, len(streams)):
    for j in range(1, streams[i].shape[0]-1):
        pt = streams[i][j]
        pt[1] = -pt[1]
        surface_points.append(tuple(pt))
        
# Create CAD boundary
edge_wire = cq.Workplane("XY").spline([tuple(x) for x in le])
edge_wire = edge_wire.add(cq.Workplane("XY").spline([tuple(x) for x in te]))

# Create CAD surface
bottom = cq.Workplane("XY").interpPlate(edge_wire, surface_points, 0)

# Add top and back as simple planes
e1 =cq.Edge.makeLine(cq.Vector(tuple(le[0])),cq.Vector(tuple(le[-1])))
e2 =cq.Edge.makeSpline([cq.Vector(tuple(x)) for x in le])
top = cq.Face.makeFromWires(cq.Wire.assembleEdges([e1, e2]))

e3 =cq.Edge.makeLine(cq.Vector(tuple(te[0])),cq.Vector(tuple(te[-1])))
e4 =cq.Edge.makeSpline([cq.Vector(tuple(x)) for x in te])
back = cq.Face.makeFromWires(cq.Wire.assembleEdges([e3, e4]))

# Create solid
waverider = cq.Solid.makeSolid(cq.Shell.makeShell([bottom.objects[0], top, back]))

# Export
cq.exporters.export(waverider, f'conical_waverider_surface_M{Mach}.step')

