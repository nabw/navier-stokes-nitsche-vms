from dolfin import *

# Define mesh
nx = 32
mesh = UnitSquareMesh(nx, nx)

# Define Function Spaces
V = VectorElement("CG", mesh.ufl_cell(), 2)
W = FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
me = MixedElement([V, W, R])
Z = FunctionSpace(mesh, me)

# Define boundary function
tol = 1e-10


def boundary(x, on_boundary):
    return on_boundary and abs(x[1] - 1) < tol


# Define Test and trial function
up = Function(Z)
u, p, rho = split(up)
v, q, lamda = TestFunctions(Z)

# Define exact solution


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
p_exact = Constant(0)
f = Constant((0, 0))

# parameter value
nu = 1/500
beta = 1
gamma = 10

n = FacetNormal(mesh)
tau = as_vector([n[1], n[0]])

h = mesh.hmin()


def D(u):
    return grad(u) + grad(u).T

# mark boundary


class OtherBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-10
        return on_boundary and (near(x[0], 0, tol) or near(x[0], 1, tol) or near(x[1], 0, tol))


other_boundary = OtherBoundary()
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)
other_boundary.mark(boundary_parts, 4)
ds = Measure("ds")[boundary_parts]

# define weak formulation
F = (nu*0.5 * inner(D(u), D(v)) * dx + inner(grad(u)*u, v) * dx - p * div(v) * dx - div(u) * q * dx + beta * dot(u, tau) * dot(v, tau) * ds(4) - Constant(nu)*dot(D(u)*dot(v, n)*n, n)*ds(4) -
     Constant(nu)*dot(D(v)*dot(u, n)*n, n)*ds(4) + rho*q*dx + lamda*p*dx + Constant(gamma) * Constant(1.0 / h) * dot(u, n) * dot(v, n) * ds(4) + p*dot(v, n)*ds(4) + q*dot(u, n)*ds(4) - dot(f, v)*dx)

bc = [DirichletBC(Z.sub(0), u_exact, boundary)]

# solve linear system
solve(F == 0, up, bc, solver_parameters={"newton_solver": {
      "linear_solver": 'mumps'}, "newton_solver": {"relative_tolerance": 1e-7}})
u, p, rho = up.split()
# p.rename("pressure","pressure")
#File("pressure.pvd") << p
u.rename("velocity", "velocity")
File("velocity.pvd") << u
