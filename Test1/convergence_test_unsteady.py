'''
Test 1: Convergence Test for Nitsche method

Consider an unsteady stationary Navier Stokes equation
 u_t -\Delta u + \nabla p + u \cdot \nabla u = 0
                              \nabla \cdot u = 0
                                           u = u_exact      on y=-1
                                         u.n = 0            on x=1,-1 and y=1
        \nu n^t D(u) \tau^i + \beta u. tau^i = 0 (i=1,2)    on x=1,-1 and y=1
 
VMS-LES Nitsche formulation is used to solve at Re = 1000 
mean of pressure = 0 imposed via Lagrange multiplier
'''

from dolfin import *
from ufl import JacobianInverse, indices
import matplotlib.pyplot as plt
set_log_level(LogLevel.ERROR)

# parameters
nu = Constant(0.001)
beta=Constant(1)
gamma=Constant(10)

T = 1
nx=  128
num_steps = 4
dt = T/num_steps

mesh = UnitSquareMesh(nx,nx)

# Function Spaces
V = VectorElement("CG", mesh.ufl_cell(), 1)
W= FiniteElement("CG", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
me = MixedElement([V, W, R])
Z= FunctionSpace(mesh,me)

# Test and trial functions
up = Function(Z)
u, p, rho  = split(up)
v, q, lamda = TestFunctions(Z)

# known data
u_exact = Expression(('sin(pi*x[0]-0.7)*sin(pi*x[1]+0.2)*cos(t)','cos(pi*x[0]-0.7)*cos(pi*x[1]+0.2)*cos(t)'), domain =mesh, degree =5, t=0)             
p_exact = Expression(('cos(t)*(sin(x[0])*cos(x[1])+(cos(1)-1)*sin(1))'), domain =mesh, degree =5, t=0)         
u_exd =  Expression(('-sin(pi*x[0]-0.7)*sin(pi*x[1]+0.2)*sin(t)','-cos(pi*x[0]-0.7)*cos(pi*x[1]+0.2)*sin(t)'), domain =mesh, degree =5, t=0) 
f1 = u_exd -nu*div(grad(u_exact))+ grad(u_exact)*(u_exact) +grad(p_exact)
f2= div(u_exact)
H = dot(u_exact,n)
ss = inner(nu*D(u_exact)*n,tau)+beta*dot(u_exact,tau)

         
def D(u):
    return grad(u) + grad(u).T

# data at previous steps   
u_n = Function(Z.sub(0).collapse())
u_n.interpolate(u_exact)
p_n = Function(Z.sub(1).collapse())
p_n.interpolate(p_exact)


n=  FacetNormal(mesh)
tau = as_vector((-n[1], n[0])) 
h = mesh.hmin()
idt = Constant(1/dt)

XI_X = JacobianInverse(mesh)
G = XI_X.T * XI_X  # Metric tensor
g = [0, 0]  # Metric vector
g[0] = XI_X[0, 0] + XI_X[1, 0] 
g[1] = XI_X[0, 1] + XI_X[1, 1] 
g = as_vector(g)

# stabilization and residuals
T_M = pow((idt*idt+inner(u_n,dot(u_n,G))+30*nu*nu*inner(G,G)),-0.5)
T_C = pow((T_M*dot(g,g)),-1)
R_M = (u -u_n)*idt + grad(u)*u_n  + grad(p) -nu*div(grad(u))-f1
r_M = grad(u_n)*u_n + grad(p_n) -nu*div(grad(u_n))-f1
R_C = div(u)

# mark boundaries
class Others(SubDomain):
      def inside(self, x, on_boundary):
          tol = 1e-10
          return on_boundary and (near(x[0],0,tol) or near(x[0],1,tol) or near(x[1],0,tol))
class Top(SubDomain):
      def inside(self,x,on_boundary):
          return on_boundary and near(x[1],1)

dirichlet_boundary = Others()
slip_boundary = Top()
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(0)
slip_boundary.mark(boundary_parts, 1)
dirichlet_boundary.mark(boundary_parts,2)
ds = Measure("ds")[boundary_parts]

#VMS-LES NITSCHE FORMULATION
F = idt*dot(v,u)*dx - idt*dot(v,u_n)*dx + nu*0.5 * inner(D(u), D(v)) * dx + inner (grad(u)*u_n,v)*dx - p * div(v) * dx - div(u) * q * dx + rho*q*dx + lamda*p*dx - dot(f1,v)*dx + q*f2*dx 

F = F + beta * dot(u, tau) * dot(v, tau) * ds(1)- nu*dot(D(u)*dot(v,n)*n, n)*ds(1)- nu*dot(D(v)*dot(u,n)*n, n)*ds(1) + p*dot(v,n)*ds(1) + q*dot(u,n)*ds(1) + gamma * Constant(1.0 / h) * dot(u, n) * dot(v, n) * ds(1) - ss*dot(v,tau)*ds(1)+ nu*dot(D(v)*H*n, n)*ds(1)- q*H*ds(1) -gamma * Constant(1.0 / h) * H * dot(v, n) * ds(1) 

F = F + inner(grad(v)*u_n,T_M*R_M)*dx - dot(grad(q),T_M*R_M)*dx + dot(div(v),T_C*R_C)*dx + inner((grad(v).T)*u_n,T_M*R_M)*dx - inner(grad(v),outer(T_M*r_M,T_M*R_M))*dx - T_C*R_C*dot(v,n)*ds(1) - q*dot(T_M*R_M,n)*ds(1) + nu*dot(D(T_M*R_M)*dot(v,n)*n, n)*ds(1) +nu*dot(D(v)*dot(T_M*R_M,n)*n, n)*ds(1) 

t=0
e_time_u =0
e_time_p =0
for z in range(num_steps):
    t = (z+1)*dt
    u_exact.t=t
    p_exact.t=t
    u_exd.t = t
    bc = [DirichletBC(Z.sub(0), u_exact, boundary_parts, 2)]
    solve(F == 0, up, bc,  solver_parameters={"newton_solver":{"linear_solver":'mumps'},"newton_solver":{"relative_tolerance":1e-7}})
    u, p, rho = up.split()
    assign(u_n,u)
    assign(p_n,p)
    
    error_H1 = abs(assemble(dot(u-u_exact,u-u_exact)*dx+inner(grad(u)-grad(u_exact),grad(u)-grad(u_exact))*dx))
    error_L2 = abs(assemble(dot(p-p_exact,p-p_exact)*dx))
    e_time_u = e_time_u + dt*error_H1
    e_time_p = e_time_p + dt*error_L2

e_time_uu = sqrt(e_time_u)
e_time_pp = sqrt(e_time_p)

print(' error of velocity =',e_time_uu)
print(' error of pressure =',e_time_pp)



