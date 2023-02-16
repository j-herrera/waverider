# Import modules
import sys
sys.path.append('../')

import numpy as np
from waverider import conical_flow as cone
from scipy.integrate import solve_ivp

from waverider.meshing_tools import points_to_ply


# Define constants
Mach = 4
gamma = 1.4
cone_angle = 20 * np.pi / 180
N_stream = 100


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
    sol = solve_ivp(stode, (0, 1000), [x_le[i], eta_le[i]], events=back, max_step=1)
    streams.append(np.vstack([sol.y[0], sol.y[1] * np.sin(alphas[i]), -sol.y[1] * np.cos(alphas[i])]).T)
   
    
# Export computed surfaces as ply
data = streams[0]
for i in range(1, N_stream):
    data = np.vstack([data, streams[i]])
points_to_ply(data, f'conical_waverider_surface_M{Mach}.ply')
