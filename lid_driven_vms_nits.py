from dolfin import *
from ufl import JacobianInverse, indices

T = 3
num_steps = 1000
dt = T/num_steps

nx = 32
mesh = UnitSquareMesh(nx, nx)

# Define Function Spaces
V = VectorElement("CG", mesh.ufl_cell(), 1)
W = FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
me = MixedElement([V, W, R])
Z = FunctionSpace(mesh, me)


tol = 1e-10


def boundary(x, on_boundary):
    return on_boundary and abs(x[1] - 1) < tol


# Define Test and trial function
up = Function(Z)
u, p, rho = split(up)
v, q, lamda = TestFunctions(Z)


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
u_in = Constant((0, 0))
u_n = project(u_in, Z.sub(0).collapse())
p_n = project(p_exact, Z.sub(1).collapse())


nu = Constant(1/1000)
beta = Constant(1)
gamma = Constant(10)


idt = Constant(1/dt)
n = FacetNormal(mesh)
tau = as_vector([n[1], - n[0]])
h = mesh.hmin()


XI_X = JacobianInverse(mesh)
G = XI_X.T * XI_X  # Metric tensor
g = [0, 0]  # Metric vector
g[0] = XI_X[0, 0] + XI_X[1, 0]
g[1] = XI_X[0, 1] + XI_X[1, 1]
g = as_vector(g)


T_M = pow((idt*idt+inner(u_n, dot(u_n, G))+30*nu*nu*inner(G, G)), -0.5)
T_C = pow((T_M*dot(g, g)), -1)
T_M_1 = project(T_M, Z.sub(1).collapse())
T_C_1 = project(T_C, Z.sub(1).collapse())
R_M = (u - u_n)*idt + grad(u)*u_n + grad(p) - nu*div(grad(u))-f
r_M = grad(u_n)*u_n + grad(p_n) - nu*div(grad(u_n))-f
R_C = div(u)


def D(u):
    return grad(u) + grad(u).T


class OtherBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-10
        return on_boundary and (near(x[0], 0, tol) or near(x[0], 1, tol) or near(x[1], 0, tol))


other_boundary = OtherBoundary()
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)
other_boundary.mark(boundary_parts, 1)
ds = Measure("ds")[boundary_parts]


F = (idt*dot(v, u)*dx - idt*dot(v, u_n)*dx + nu*0.5 * inner(D(u), D(v)) * dx + inner(grad(u)*u_n, v)*dx - p * div(v) * dx - div(u) * q * dx + beta * dot(u, tau) * dot(v, tau) * ds(1) - nu*dot(D(u)*dot(v, n)*n, n)*ds(1) - nu*dot(D(v)*dot(u, n)*n, n)*ds(1) + rho*q*dx + lamda*p*dx + gamma * Constant(1.0 / h)
     * dot(u, n) * dot(v, n) * ds(1) + p*dot(v, n)*ds(1) + q*dot(u, n)*ds(1) + inner(grad(v)*u_n, T_M*R_M)*dx - dot(grad(q), T_M*R_M)*dx + dot(div(v), T_C*R_C)*dx + inner((grad(v).T)*u_n, T_M*R_M)*dx - inner(grad(v), outer(T_M*r_M, T_M*R_M))*dx - T_C*R_C*dot(v, n)*ds(1) - q*dot(T_M*R_M, n)*ds(1) - dot(f, v)*dx)

bc = [DirichletBC(Z.sub(0), u_exact, boundary)]
t = 0

for z in range(num_steps):
    t = (z+1)*dt
    u_exact.t = t
    p_exact.t = t
    solve(F == 0, up, bc, solver_parameters={"newton_solver": {
          "linear_solver": 'mumps'}, "newton_solver": {"relative_tolerance": 1e-7}})
    u, p, rho = up.split()
    assign(u_n, u)
    assign(p_n, p)
# T_M_1.rename("S_M","S_M")
#File("s_m.pvd") << T_M_1
# T_C_1.rename("S_C","S_C")
#File("s_c.pvd") << T_C_1

# p.rename("pressure","pressure")
#File("pressure.pvd") << p
u.rename("velocity", "velocity")
File("velocity.pvd") << u
