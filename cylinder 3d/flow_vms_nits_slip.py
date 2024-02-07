from dolfin import *
from ufl import JacobianInverse, indices
from dolfin import project
import numpy
import meshio

T = 10
num_steps = 50
dt = T/num_steps

dim   = 3        
msh = meshio.read("test.msh")
def create_mesh(mesh, cell_type, outname, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={outname:[cell_data]})
    return out_mesh

if dim==2:

    # Create and save one file for the mesh, and one file for the facets
    triangle_mesh = create_mesh(msh, "triangle", "subdomains",prune_z=True)
    line_mesh     = create_mesh(msh, "line", "boundaries",prune_z=True)

    meshio.write("test_domain.xdmf", triangle_mesh)
    
    meshio.write("test_boundaries.xdmf", line_mesh)

else:

    tetra_mesh    = create_mesh(msh, "tetra","subdomains",prune_z=False)
    triangle_mesh = create_mesh(msh, "triangle","boundaries",prune_z=False)
    #
    meshio.write("test_domain.xdmf", tetra_mesh)
    meshio.write("test_boundaries.xdmf", triangle_mesh)

# Optimization options:
parameters["form_compiler"]["cpp_optimize"]       = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
parameters["form_compiler"]["optimize"]           = True
#parameters["refinement_algorithm"] = "plaza_with_parent_facets"
#parameters["ghost_mode"]           = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 4

# reading the mesh made using gmsh (see ../mesh ):
mesh = Mesh()
with XDMFFile("test_domain.xdmf") as infile:
     infile.read(mesh)
mvc  = MeshValueCollection("size_t", mesh, dim-1)
with XDMFFile("test_boundaries.xdmf") as infile:
     infile.read(mvc, 'boundaries')
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)
#
mvc2 = MeshValueCollection("size_t", mesh, dim)
with XDMFFile("test_domain.xdmf") as infile:
     infile.read(mvc2, 'subdomains')
domains = cpp.mesh.MeshFunctionSizet(mesh, mvc2)
  
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

V = VectorElement("CG", mesh.ufl_cell(), 1)
W= FiniteElement("CG", mesh.ufl_cell(), 1)
#R = FiniteElement("Real", mesh.ufl_cell(), 0)
me = MixedElement([V, W])
#me = MixedElement([V, W, R])
Z= FunctionSpace(mesh,me)
print(Z.dim())

up = Function(Z)
#u, p, rho  = split(up)
#v, q, lamda = TestFunctions(Z)
u, p  = split(up)
v, q = TestFunctions(Z)

h  = mesh.hmin()
nu = Constant(1/10)
beta=1
gamma = 10

idt = Constant(1/dt)
u_exact = Expression(("(16*2.25*x[1]*x[2]*sin(0.125*pi*t)*(0.41-x[1])*(0.41-x[2]))/(0.41*0.41*0.41*0.41)", "0", "0"), domain =mesh, degree=6, t=0)

f1 = Constant((0,0,0))
f2 = Constant(0)

u_in = Function(Z.sub(0).collapse())
p_in = Function(Z.sub(1).collapse())

u_n = project(u_in,Z.sub(0).collapse(), solver_type="mumps", form_compiler_parameters=None)
p_n = project(p_in,Z.sub(1).collapse(),solver_type="mumps", form_compiler_parameters=None)

n = FacetNormal(mesh)
tau1 = as_vector([-n[1], n[0], 0])
tau2 = as_vector([0, 0 , 1])

def D(v):
    return (grad(v) + grad(v).T)
    
XI_X = JacobianInverse(mesh)
G = XI_X.T * XI_X  # Metric tensor
g = [0, 0, 0]  # Metric vector
g[0] = XI_X[0, 0] + XI_X[1, 0] + XI_X[2, 0]
g[1] = XI_X[0, 1] + XI_X[1, 1] + XI_X[2, 1]
g[2] = XI_X[0, 2] + XI_X[1, 2] + XI_X[2, 2]
g = as_vector(g)

T_M = pow((idt*idt+inner(u_n,dot(u_n,G))+30*nu*nu*inner(G,G)),-0.5)
T_C = pow((T_M*dot(g,g)),-1)
R_M = (u -u_n)*idt + dot(u_n,nabla_grad(u)) + grad(p) -nu*div(grad(u))-f1
r_M =  dot(u_n,nabla_grad(u_n)) + grad(p_n) -nu*div(grad(u_n))-f1
R_C = div(u)

F = (idt*dot(v,u)*dx - idt*dot(v,u_n)*dx + nu*0.5 * inner(D(u), D(v)) * dx + inner (grad(u)*u_n,v)*dx - p * div(v) * dx - div(u) * q * dx - dot(f1,v)*dx + q*f2*dx  + beta * dot(u, tau1) * dot(v, tau1) * ds(3) + beta * dot(u, tau2) * dot(v, tau2) * ds(3) - nu*dot(D(u)*dot(v,n)*n, n)*ds(3)- nu*dot(D(v)*dot(u,n)*n, n)*ds(3) + p*dot(v,n)*ds(3) + q*dot(u,n)*ds(3) + gamma * Constant(1.0 / h) * dot(u, n) * dot(v, n) * ds(3) + inner(grad(v)*u_n,T_M*R_M)*dx - dot(grad(q),T_M*R_M)*dx + dot(div(v),T_C*R_C)*dx + inner((grad(v).T)*u_n,T_M*R_M)*dx - inner(grad(v),outer(T_M*r_M,T_M*R_M))*dx - T_C*R_C*dot(v,n)*ds(3) - q*dot(T_M*R_M,n)*ds(3) + nu*dot(D(T_M*R_M)*dot(v,n)*n, n)*ds(3) +nu*dot(D(v)*dot(T_M*R_M,n)*n, n)*ds(3) )
#F += rho*q*dx + lamda*p*dx 



u_n.rename("velocity_slip","velocity_slip")
p_n.rename("pressure_slip","pressure_slip")
file_u = File("velocity_slip.pvd")
file_p = File("pressure_slip.pvd")
for z in range(num_steps):
    t = (z+1)*dt
    print("Time", t, (z+1)/num_steps)
    u_exact.t=t
    bc1 = DirichletBC(Z.sub(0), Constant((0,0,0)), boundaries, 41)
    bc2 = DirichletBC(Z.sub(0), Constant((0,0,0)), boundaries, 42)
    bc3 = DirichletBC(Z.sub(0), Constant((0,0,0)), boundaries, 43)
    bc4 = DirichletBC(Z.sub(0), Constant((0,0,0)), boundaries, 44)
    bc5 = DirichletBC(Z.sub(0), u_exact, boundaries, 1)
    bc = [bc1, bc2, bc3, bc4, bc5]
    solve(F == 0, up, bc, solver_parameters={"newton_solver":{"linear_solver":'mumps'},"newton_solver":{"relative_tolerance":1e-6}})
    #u, p, rho = up.split()
    u, p = up.split()
    assign(u_n,u)
    assign(p_n,p)
    
    file_u << u, t
    file_p << p, t



