'''
Test 3: Flow past through a circular cylinder: No slip case

Consider a unsteady Navier Stokes equation
u_t + \Delta u + \nabla p + u \cdot \nabla u = f
                          \nabla \cdot u = 0
 Boundary coundtions:
 u = 0 is imposed on  lateral walls and circular cylinder
 Do nothing bc on outflow
 u_exact = (16 U_m sin(pi t /8)yz(H-y)(H-z)/H^4,0,0) where H=0.41m and U_m = 2.25m/sec on inflow
 
 VMS-LES formulation is used to solve at Re = 1000

'''
from dolfin import *
from ufl import JacobianInverse, indices
from dolfin import project
# Optimization options:
parameters["form_compiler"]["cpp_optimize"]       = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
parameters["form_compiler"]["optimize"]           = True
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters["ghost_mode"]           = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 8     

dim=3
# reading the mesh made using gmsh (see ../mesh ):
mesh = Mesh()
with XDMFFile("mesh/test_domain.xdmf") as infile:
     infile.read(mesh)
mvc  = MeshValueCollection("size_t", mesh, dim-1)
with XDMFFile("mesh/test_boundaries.xdmf") as infile:
     infile.read(mvc, 'boundaries')
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
#
mvc2 = MeshValueCollection("size_t", mesh, dim)
with XDMFFile("mesh/test_domain.xdmf") as infile:
     infile.read(mvc2, 'subdomains')
domains = cpp.mesh.MeshFunctionSizet(mesh, mvc2)
  
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
T = 1
num_steps = 4
dt = T/num_steps

# Function spaces
V = VectorElement("CG", mesh.ufl_cell(), 1)
W= FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
me = MixedElement([V, W, R])
Z= FunctionSpace(mesh,me)

# Test and Trial functions
up = Function(Z)
u, p, rho  = split(up)
v, q, lamda = TestFunctions(Z)

# parameters
h  = mesh.hmin()
nu = Constant(0.000099)
beta=0
gamma = 10
idt = Constant(1/dt)

# known data
u_exact = Expression(("(16*2.25*x[1]*x[2]*sin(0.125*pi*t)*(0.41-x[1])*(0.41-x[2]))/(0.41*0.41*0.41*0.41)", "0", "0"), domain =mesh, degree=6, t=0)
f1 = Constant((0,0,0))
f2 = Constant(0)

# u and p at previous time steps
u_in = Function(Z.sub(0).collapse())
p_in = Function(Z.sub(1).collapse())

u_n = project(u_in,Z.sub(0).collapse(), solver_type="mumps", form_compiler_parameters=None)
p_n = project(p_in,Z.sub(1).collapse(),solver_type="mumps", form_compiler_parameters=None)

def D(v):
    return (grad(v) + grad(v).T)
    
XI_X = JacobianInverse(mesh)
G = XI_X.T * XI_X  # Metric tensor
g = [0, 0, 0]      # Metric vector
g[0] = XI_X[0, 0] + XI_X[1, 0] + XI_X[2, 0]
g[1] = XI_X[0, 1] + XI_X[1, 1] + XI_X[2, 1]
g[2] = XI_X[0, 2] + XI_X[1, 2] + XI_X[2, 2]
g = as_vector(g)

# stabilization and residuals
T_M = pow((idt*idt+inner(u_n,dot(u_n,G))+30*nu*nu*inner(G,G)),-0.5)
T_C = pow((T_M*dot(g,g)),-1)
R_M = (u -u_n)*idt + grad(u)*u_n  + grad(p) -nu*div(grad(u))-f1
r_M = grad(u_n)*u_n + grad(p_n) -nu*div(grad(u_n))-f1
R_C = div(u)

# VMS-LES FORMULATION
F = idt*dot(v,u)*dx - idt*dot(v,u_n)*dx + nu*0.5 * inner(D(u), D(v)) * dx + inner(dot(u_n,nabla_grad(u)), v) * dx - p * div(v) * dx - div(u) * q * dx - dot(f1,v)*dx + f2*q*dx 
F = F +  inner(dot(u_n,nabla_grad(v)),T_M*R_M)*dx - dot(grad(q),T_M*R_M)*dx + dot(div(v),T_C*R_C)*dx + inner(dot(u_n,nabla_grad(v).T),T_M*R_M)*dx -  inner(grad(v),outer(T_M*r_M,T_M*R_M))*dx  

for z in range(num_steps):
    t = (z+1)*dt
    u_exact.t=t
    bc1 = DirichletBC(Z.sub(0), Constant((0,0,0)), boundaries, 41)
    bc2 = DirichletBC(Z.sub(0), Constant((0,0,0)), boundaries, 42)
    bc3 = DirichletBC(Z.sub(0), Constant((0,0,0)), boundaries, 43)
    bc4 = DirichletBC(Z.sub(0), Constant((0,0,0)), boundaries, 44)
    bc5 = DirichletBC(Z.sub(0), u_exact, boundaries, 1)
    bc6 = DirichletBC(Z.sub(0), Constant((0,0,0)), boundaries, 3)
    bc = [bc1, bc2, bc3, bc4, bc5, bc6]
    solve(F == 0, up, bc, solver_parameters={"newton_solver":{"linear_solver":'mumps'},"newton_solver":{"relative_tolerance":1e-6}})
    u, p, rho = up.split()
    assign(u_n,u)
    assign(p_n,p)

# Saving data
u.rename("velocity_noslip","velocity_noslip")
File("velocity_noslip.pvd") << u
p.rename("pressure_noslip","pressure_noslip")
File("pressure_noslip.pvd") << p



