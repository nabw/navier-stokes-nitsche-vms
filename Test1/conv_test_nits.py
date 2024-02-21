'''
Test 1: Convergence Test for Nitsche method

Consider an stationary Navier Stokes equation
     -\Delta u + \nabla p + u \cdot \nabla u = 0
                              \nabla \cdot u = 0
                                           u = u_exact      on y=-1
                                         u.n = 0            on x=1,-1 and y=1
        \nu n^t D(u) \tau^i + \beta u. tau^i = 0 (i=1,2)    on x=1,-1 and y=1
 
Nitsche formulation is used to solve at Re = 1 
mean of pressure = 0 imposed via Lagrange multiplier
'''

from dolfin import *
set_log_level(LogLevel.ERROR)
# Optimization options:
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
parameters["form_compiler"]["optimize"] = True
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 8

# Space dimension
dim = 2

# reading the mesh made using gmsh (see ../mesh ):
mesh = Mesh()
with XDMFFile("mesh/square_domain.xdmf") as infile:
    infile.read(mesh)
mvc = MeshValueCollection("size_t", mesh, dim-1)
with XDMFFile("mesh/square_boundaries.xdmf") as infile:
    infile.read(mvc, 'boundaries')
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

mvc2 = MeshValueCollection("size_t", mesh, dim)
with XDMFFile("mesh/square_domain.xdmf") as infile:
    infile.read(mvc2, 'subdomains')
domains = cpp.mesh.MeshFunctionSizet(mesh, mvc2)

h_mesh = []
ep_L2 = []
eu_L2 = []
eu_sH1 = []

nmax = 3  # Number of uniform adapted steps

for iterAdapt in range(nmax):
    H = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    R = FiniteElement("Real", mesh.ufl_cell(), 0)
    V = FunctionSpace(mesh, MixedElement([H, Q]))
    print(V.dim())
    up = Function(V)
    u, p, rho = split(up)
    v, q, lamda = TestFunctions(V)

    n = FacetNormal(mesh)
    t = as_vector([n[1], -n[0]])
    h = mesh.hmin()

    # parameters
    nu = Constant(1)
    beta = 10
    gamma = 1

    def epsilon(v):
        return (grad(v) + grad(v).T)

    def u_n(v):
        return inner(v, n)

    def u_t(v):
        return inner(v, t)

   # Analytical solution
    u_exac = Expression(
        ('2.0*x[1]*(1.0 -x[0]*x[0])', '-2.0*x[0]*(1.0-x[1]*x[1])'), domain=mesh, degree=3)
    p_exac = Expression('(2*x[0]-1)*(2*x[1]-1)-1', domain=mesh, degree=2)

  # Data:
    f = -nu*div(grad(u_exac)) + grad(p_exac) + grad(u_exac)*(u_exac)
    g = Constant(0.0)
    ss = inner(nu*epsilon(u_exac)*n, t) + beta*u_t(u_exac)

    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    # NITSCHE FORMULATION
    F = (0.5*nu*inner(epsilon(u), epsilon(v)) - div(v)*p - q*div(u) +
         inner(grad(u)*u, v))*dx + rho*q*dx + lamda*p*dx - inner(f, v)*dx
    F = F + beta * u_t(u) * u_t(v) * ds(1) - nu*dot(epsilon(u)*u_n(v)*n, n)*ds(1)-nu*dot(epsilon(v)*u_n(u)*n, n)*ds(1) + p*u_n(v)*ds(
        1) + q*u_n(u)*ds(1) + gamma*u_n(u)*u_n(v)*ds(1) - nu*dot(epsilon(v)*g*n, n)*ds(1)+q*g*ds(1) + gamma*g*u_n(v)*ds(1) + ss*u_t(v)*ds(1)

    # boundary
    bc = DirichletBC(V.sub(0), u_exac, boundaries, 2)

    # solve
    solve(F == 0, up, bc, solver_parameters={"newton_solver": {
          "linear_solver": 'mumps'}, "newton_solver": {"relative_tolerance": 1e-7}})
    (u, p, rho) = up.split(True)

    # Error's computation:
    error_p_L2 = 0.0
    error_u_L2 = 0.0
    error_u_sH1 = 0.0
    error_slip = 0.0

    error_p_L2 = errornorm(p_exac, p, norm_type="L2", degree_rise=6, mesh=mesh)
    error_u_L2 = errornorm(u_exac, u, norm_type="L2", degree_rise=6, mesh=mesh)
    error_u_sH1 = errornorm(u_exac, u, norm_type="H10",
                            degree_rise=6, mesh=mesh)
    # ||u.n -g||_{0,\Gamma_s}
    error_slip = sqrt(abs(assemble(u_n(u)*(u_n(u))*ds(1))))

    h_mesh.append(mesh.hmax())
    ep_L2.append(error_p_L2)
    eu_L2.append(error_u_L2)
    eu_sH1.append(error_u_sH1)

    O_p_L2 = 0.0
    O_u_L2 = 0.0
    O_u_sH1 = 0.0
    # h    dof     Nelem     e_p     e_u     error  Eta  Effectivity
    if iterAdapt > 0:
        O_p_L2 = ln(ep_L2[iterAdapt]/ep_L2[iterAdapt-1]) / \
            ln(h_mesh[iterAdapt]/h_mesh[iterAdapt-1])     #
        O_u_L2 = ln(eu_L2[iterAdapt]/eu_L2[iterAdapt-1]) / \
            ln(h_mesh[iterAdapt]/h_mesh[iterAdapt-1])     # convergence orders
        O_u_sH1 = ln(eu_sH1[iterAdapt]/eu_sH1[iterAdapt-1]) / \
            ln(h_mesh[iterAdapt]/h_mesh[iterAdapt-1])   #

    # Uniform adapted mesh:
    mesh = adapt(mesh)
    boundaries = adapt(boundaries, mesh)
    domains = adapt(domains, mesh)

print("%f, %f, %f, %f, %f, %f, %f, %f \r\n" % (
    h_mesh[iterAdapt], error_p_L2, O_p_L2, error_u_L2, O_u_L2, error_u_sH1, O_u_sH1, error_slip))
