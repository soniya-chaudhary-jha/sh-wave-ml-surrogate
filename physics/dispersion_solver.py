import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import fsolve
from physics.material_constants import C66, C44, C44_ve, rho1, rho2

def dispersion_equation(c, k, H, alpha, eta):
    beta1 = np.sqrt(C66 / rho1)

    # simplified proxy of full equation (stable ML data generator)
    term1 = np.tan(k * H * np.sqrt((c/beta1)**2 - 1))
    damping = eta * 0.01
    hetero = alpha * 0.05

    term2 = (C44_ve / C44) * (1 + damping + hetero)

    return term1 - term2

def solve_phase_velocity(k, H, alpha, eta):
    c0 = np.sqrt(C66 / rho1) * 1.2

    try:
        c_solution = fsolve(
            dispersion_equation, c0, args=(k, H, alpha, eta)
        )[0]
        return np.abs(c_solution)
    except:
        return np.nan
