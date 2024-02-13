## **Repository Description**

This repository contains the code utilized to generate the results presented in the paper titled 'Nitsche method for Navier-Stokes equations with slip boundary conditions: convergence analysis and VMS-LES stabilization.'

## **Models Provided**

Two models are included in this repository:

- **Stationary Navier-Stokes Equation with Slip Boundary Conditions:** This model is suitable for low Reynolds numbers. It employs the Nitsche method for solution.

- **Non-stationary Navier-Stokes Equation with Slip Boundary Conditions:** This model includes a non-stationary term and is applicable for high Reynolds numbers. The VMS-LES Nitsche scheme is employed for stabilization.

## **Tests Conducted**

Three tests are conducted in this work:

1. **Test 1: Convergence Test**
   - Scripts: `conv_test_nits.py` and `convergence_test_unsteady.py`

2. **Test 2: Lid Driven Cavity Test**
   - Scripts: `lid_driven_steady_noslip.py`, `lid_driven_steady_slip.py`, `lid_driven_unsteady_noslip.py`, and `lid_driven_unsteady_slip.py`

3. **Test 3: Flow Past a Circular Cylinder**
   - Scripts: `flow_past_cylinder_noslip.py` and `flow_past_cylinder_slip.py`

## **Dependencies**

The code is dependent on FEniCS 2019.1.0 with Python 3. Please ensure that this dependency is satisfied before running the scripts.
