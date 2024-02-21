'''
Test 2: Lid driven cavity
domain: Unit Square
Consider an unsteady Navier Stokes equation
u_t  -\Delta u + \nabla p + u \cdot \nabla u = 0
                              \nabla \cdot u = 0
                                           u = u_exact      on y=1
                                         u.n = 0            on x=0,1 and y=0
        \nu n^t D(u) \tau^i + \beta u. tau^i = 0 (i=1,2)    on x=0,1 and y=0
 
VMS-LES Nitsche formulation is used to solve at Re = 1000 and 5000
mean of pressure = 0 imposed via Lagrange multiplier
'''

from dolfin import *
from ufl import JacobianInverse, indices

T = 50
num_steps = 100
dt = T/num_steps

nx = 128
mesh = UnitSquareMesh(nx, nx)

# Function Spaces
V = VectorElement("CG", mesh.ufl_cell(), 1)
W = FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
me = MixedElement([V, W, R])
Z = FunctionSpace(mesh, me)

# Test and trial functions
up = Function(Z)
u, p, rho = split(up)
v, q, lamda = TestFunctions(Z)

# boundary data on y=1


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
f1 = Constant((0, 0))
f2 = Constant(0)

# data at previous step
u_in = Function(Z.sub(0).collapse())
p_in = Function(Z.sub(1).collapse())

u_n = Function(Z.sub(0).collapse())
u_n.interpolate(u_in)
p_n = Function(Z.sub(1).collapse())
p_n.interpolate(p_in)

# parameters
nu = Constant(1/1000)
beta = Constant(1)
gamma = Constant(10)

idt = Constant(1/dt)
n = FacetNormal(mesh)
tau = as_vector([-n[1], n[0]])
h = mesh.hmin()


XI_X = JacobianInverse(mesh)
G = XI_X.T * XI_X  # Metric tensor
g = [0, 0]  # Metric vector
g[0] = XI_X[0, 0] + XI_X[1, 0]
g[1] = XI_X[0, 1] + XI_X[1, 1]
g = as_vector(g)

# stabilization and residual
T_M = pow((idt*idt+inner(u_n, dot(u_n, G))+30*nu*nu*inner(G, G)), -0.5)
T_C = pow((T_M*dot(g, g)), -1)
R_M = (u - u_n)*idt + grad(u)*u_n + grad(p) - nu*div(grad(u))-f1
r_M = grad(u_n)*u_n + grad(p_n) - nu*div(grad(u_n))-f1
R_C = div(u)


def D(u):
    return grad(u) + grad(u).T

# mark boundaries


class Others(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-10
        return on_boundary and (near(x[0], 0, tol) or near(x[0], 1, tol) or near(x[1], 0, tol))


class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1)


slip_boundary = Others()
dirichlet_boundary = Top()
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)
slip_boundary.mark(boundary_parts, 1)
dirichlet_boundary.mark(boundary_parts, 2)
ds = Measure("ds")[boundary_parts]

# VMS-LES NITSCHE FORMULATION
F = idt*dot(v, u)*dx - idt*dot(v, u_n)*dx + nu*0.5 * inner(D(u), D(v)) * dx + inner(grad(u)*u_n,
                                                                                    v)*dx - p * div(v) * dx - div(u) * q * dx + rho*q*dx + lamda*p*dx - dot(f1, v)*dx + q*f2*dx
F = F + beta * dot(u, tau) * dot(v, tau) * ds(1) - Constant(nu)*dot(D(u)*dot(v, n)*n, n)*ds(1) - Constant(nu)*dot(D(v)*dot(u, n)
                                                                                                                  * n, n)*ds(1) + Constant(gamma) * Constant(1.0 / h) * dot(u, n) * dot(v, n) * ds(1) + p*dot(v, n)*ds(1) + q*dot(u, n)*ds(1)
F = F + inner(grad(v)*u_n, T_M*R_M)*dx - dot(grad(q), T_M*R_M)*dx + dot(div(v), T_C*R_C) * \
    dx + inner((grad(v).T)*u_n, T_M*R_M)*dx - \
    inner(grad(v), outer(T_M*r_M, T_M*R_M))*dx
F = F + nu*dot(D(T_M*R_M)*dot(v, n)*n, n)*ds(1) + nu*dot(D(v)*dot(T_M*R_M, n)
                                                         * n, n)*ds(1) - T_C*R_C*dot(v, n)*ds(1) - q*dot(T_M*R_M, n)*ds(1)

t = 0
for z in range(num_steps):
    t = (z+1)*dt
    u_exact.t = t
    bc = [DirichletBC(Z.sub(0), u_exact, boundary_parts, 2)]
    solve(F == 0, up, bc, solver_parameters={"newton_solver": {
          "linear_solver": 'mumps'}, "newton_solver": {"relative_tolerance": 1e-7}})
    u, p, rho = up.split()
    assign(u_n, u)
    assign(p_n, p)

# saving data
u.rename("velocity_slip", "velocity_slip")
File("velocity_slip.pvd") << u
