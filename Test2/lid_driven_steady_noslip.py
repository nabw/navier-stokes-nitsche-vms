'''
Test 2: Lid driven cavity
domain: Unit Square
Consider an stationary Navier Stokes equation
     -\Delta u + \nabla p + u \cdot \nabla u = 0
                              \nabla \cdot u = 0
                                           u = u_exact      on y=1
                                           u = 0            on x=0,1 and y=0
 
Nitsche formulation is used to solve at Re = 1 and 500
mean of pressure = 0 imposed via Lagrange multiplier
'''

from dolfin import *
set_log_level(LogLevel.ERROR)

# Define mesh
nx=64
mesh = UnitSquareMesh(nx,nx)

# Function Spaces
V = VectorElement("CG", mesh.ufl_cell(), 2)
W= FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
me = MixedElement([V, W, R])
Z= FunctionSpace(mesh,me)

# Test and trial function
up = Function(Z)
u, p, rho  = split(up)
v, q, lamda = TestFunctions(Z)

# boundary data on y =1
class BoundaryCondition(UserExpression):
      def eval(self, values, x):
          if near(x[1], 1.0) and 0.0 <= x[0] <= 0.1:
             values[0] = 10.0 * x[0]
             values[1] = 0.0
          elif near(x[1], 1.0) and 0.1 <= x[0] <= 0.9:
              values[0] = 1.0
              values[1] = 0.0
          elif near(x[1], 1.0) and 0.9 <= x[0] <= 1.0:
               values[0] = 10.0 - 10.0 * x[0]
               values[1] = 0.0
          else:
              values[0] = 0.0
              values[1] = 0.0
    
      def value_shape(self):
          return (2,)
u_exact = BoundaryCondition(degree=1)             

# known data
f = Constant((0,0))

# parameters
nu = Constant(1/500)
beta=1
gamma=10

def D(u):
    return grad(u) + grad(u).T

# mark boundaries
class Others(SubDomain):
      def inside(self, x, on_boundary):
          tol = 1e-10
          return on_boundary and (near(x[0],0,tol) or near(x[0],1,tol) or near(x[1],0,tol))
class Top(SubDomain):
      def inside(self,x,on_boundary):
          return on_boundary and near(x[1],1)

noslip_boundary = Others()
dirichlet_boundary = Top()

boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)

noslip_boundary.mark(boundary_parts, 1)
dirichlet_boundary.mark(boundary_parts,2)

ds = Measure("ds")[boundary_parts]

# Weak formulation
F = nu*0.5 * inner(D(u), D(v)) * dx + inner(grad(u)*u, v) * dx - p * div(v) * dx - div(u) * q * dx  - dot(f,v)*dx 

bc = [DirichletBC(Z.sub(0), u_exact, boundary_parts, 2), DirichletBC(Z.sub(0), Constant((0,0)), boundary_parts, 1)]

# solve linear system
solve(F == 0, up, bc, solver_parameters={"newton_solver":{"linear_solver":'mumps'},"newton_solver":{"relative_tolerance":1e-7}})
u, p, rho = up.split()

# saving data
u.rename("velocity_noslip","velocity_noslip")
File("velocity_noslip.pvd") << u
