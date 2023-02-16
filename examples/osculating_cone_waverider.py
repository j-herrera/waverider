# Import modules
import sys
sys.path.append('../')

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import root_scalar
from waverider import conical_flow as cone
from scipy.integrate import solve_ivp

from waverider.meshing_tools import points_to_ply


# Define constants
Mach = 4
gamma = 1.4
cone_angle = 20 * np.pi / 180
shock_angle = cone.shock_angle(Mach, cone_angle, gamma)[0]
N_stream = 100
x_back = 60

# Define CHS for shock line and trailing edge
x_s = np.array([0, 28])
y_s = np.array([-25, -10])
dxdy_s = np.array([0, 1.4])

x_te = np.array([0, 28])
y_te = np.array([-5, -10])
dxdy_te = np.array([0, 0])

chs_s = CubicHermiteSpline(x_s, y_s, dxdy_s)

chs_te = CubicHermiteSpline(x_te, y_te, dxdy_te)


# Support functions
def n(x, chs): # Normal
    v = np.vstack([-chs.derivative()(x), np.ones(np.atleast_1d(x).size)]).T
    v = v / np.sqrt(np.sum(v**2, 1))[:, None]
    
    return v
    

def k(x, chs): #Curvature
    dy = chs.derivative()
    ddy = dy.derivative()
    
    k = np.abs(ddy(x)) / pow(1 + dy(x)**2, 3/2)
    
    return k


def f(x, x0, chs_b, chs_t): # Osculating plane intersection
    ns = n(x0, chs_b)
    
    return chs_t(x) - chs_b(x0) - ns[:, 1] / ns[:, 0] * (x - x0)


# Propagate streamlines from the leading edge through the conical flow
[Vr, Vt] = cone.cone_field(Mach, cone_angle, shock_angle, gamma)

def stode(t, x, y_max):
    th = np.arctan(x[1] / x[0])
    
    dxdt = np.zeros(2)
    
    dxdt[0] = Vr(th) * np.cos(th) - np.sin(th) * Vt(th)
    dxdt[1] = Vr(th) * np.sin(th) + np.cos(th) * Vt(th)
    
    return dxdt
    
def back(t, y, y_max):
    return y[0] - y_max

back.terminal = True


x_sample = np.linspace(x_s[0], x_s[-1], N_stream) 

n_sample = n(x_sample, chs_s)
k_sample = k(x_sample, chs_s)

streams = []
for i in range(N_stream):
    sol = root_scalar(f, bracket=(x_te[0]-2, x_te[-1]+2), args=(x_sample[i], chs_s, chs_te)) 
    
    p2 = np.array([x_sample[i], chs_s(x_sample[i])])
    p1 = np.array([sol.root, chs_te(sol.root)]) 
    base = np.sqrt(np.sum((p1-p2)**2))
    
    r = 1 / k_sample[i]
    
    cone_apex = np.array([x_back - r / np.tan(shock_angle), p2[0] + r * n_sample[i][0], p2[1] + r * n_sample[i][1]])
    
    x_le = r / np.tan(shock_angle) - base / np.tan(shock_angle)
    eta_le = r - base
    
    alpha = np.arctan2(-n_sample[i][0], n_sample[i][1])
    
    sol = solve_ivp(stode, (0, 1000), [x_le, eta_le], events=back, args=(r / np.tan(shock_angle),), max_step=1)
    
    stream = np.vstack([sol.y[0], sol.y[1] * np.sin(alpha), -sol.y[1] * np.cos(alpha)]).T
    
    streams.append(stream + cone_apex[None, :])
   
    
# Export computed surfaces as ply
data = streams[0]
for i in range(1, N_stream):
    data = np.vstack([data, streams[i]])
points_to_ply(data, f'osculating_cone_waverider_surface_M{Mach}.ply')

