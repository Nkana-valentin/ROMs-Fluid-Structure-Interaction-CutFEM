"""
Encapsulated CutFEM Fluid-Structure Interaction Solver
Based on original code by H. v. Wahl (20.08.2020, updated 21.09.2020)
"""
from netgen.geom2d import SplineGeometry
from ngsolve import *
from xfem import *
from xfem.lsetcurv import *
from CutFEM_utilities import CheckElementHistory, UpdateMarkers, AddIntegratorsToForm
from Solvers import CutFEM_QuasiNewton
from math import ceil, pi
import time
import pickle
import os



class CutFEMProblem:
    """
    Main CutFEM problem class for fluid-structure interaction
    __author__ = H. v. Wahl
    __date__ = 20.08.2020
    __update__ = 21.09.2020
    Non-stationary test case considered in Wahl, Richter, Frei. The 
    motion of the ball is prescribed analytically.
    Rewritten by Valentin Nkana Ngan for improved modularity, readability and functionality
    for reduced order modeling and POD applications.
    """
    
    def __init__(self, args=None):
        """
        Initialize the CutFEM problem with given command line arguments
        
        Parameters:
        -----------
        args : argparse.Namespace
            Command line arguments containing h and dti
        """
        # Store arguments
        self.args = args
        
        # Set NGSolve parameters
        ngsglobals.msg_level = 1
        SetHeapSize(10000000)
        SetNumThreads(24)
        
        # Start timing
        self.start_time = time.time()
        
        # -------------------------------- PARAMETERS ---------------------------------
        # Fluid problem
        self.density_fl = 1141  # Density of fluid (kg/m^3)
        self.viscosity_dyn = 0.008  # Fluid viscosity (kg/m/s)
        
        # Solid Problem
        self.diam_ball = 0.022  # Diameter of ball (m)
        
        # Mesh parameters
        self.lowerleft = (-0.055, 0)
        self.upperright = (0.055, 0.2)  # 2D r+/z - Domain (m)
        self.h_max = args.h  # Mesh size (m)
        self.inner_factor = 4  # Inner region with h/fac
        self.bottom_factor = 1  # Inner bottom edge w/ h/fac
        self.offline_completed = False  # Run offline (no output, no timing)
        
        # Temporal parameters
        self.t_end = 2.5  # Time to run until
        self.dt_inv = args.dti  # Inverse time-step
        
        # Discretisation parameters
        self.k = 3  # Order of velocity space
        self.c_delta = 4  # Ghost-strip width param.
        self.gamma_n = 100  # Nitsche parameter
        self.gamma_s = 0.1  # Ghost-penalty stability
        self.gamma_e = 0.01  # Ghost-penalty extension
        
        # Solver parameters
        self.pReg = 1e-8  # Pressure regularisation
        self.inverse = "pardiso"  # Sparse direct solver
        self.maxit_newt = 15  # Maximum Newton steps
        self.tol_newt = 1e-10  # Residual tolerance
        self.jacobi_tol_newt = 0.1  # Jacobian update tolerance
        self.compile_flag = True  # Compile forms
        self.wait_compile = True  # Complete compile first
        
        # Output
        self.out_dir = "output/"  # Output Directory
        self.out_file = "Output_TimeDepTest_diam{}mu{}rho{}_iso-hmax{}dtinv{}BDF2.txt".format(
            self.diam_ball, self.viscosity_dyn, self.density_fl, self.h_max, self.dt_inv)
        
        # VTK output
        self.vtk_out = True  # Write VTK output
        self.vtk_subdiv = 1  # Nr subdivisions for vtk
        self.vtk_freq = self.dt_inv  # Nr. vtks per time unit
        self.vtk_dir = self.out_dir + "vtk_TimeDepTest/iso-hmax{}dtinv{}BDF2/".format(
            self.h_max, self.dt_inv)
        self.vtk_file = "BallTest-diam{}mu{}rho{}".format(
            self.diam_ball, self.viscosity_dyn, self.density_fl)
        
        # Determine inner region
        self.ur_inner = (self.diam_ball * 2 / 3, self.upperright[1])
        
        # ----------------------------------- DATA ------------------------------------
        self.velmax = 0.016
        
        # Snapshots storage for POD (optional)
        self.collect_snapshots = False
        self.snapshots_velocity = []
        self.snapshots_pressure = []
        self.snapshot_times = []
        self.snapshot_stride = 1
        
        # Initialize storage
        self.mesh = None
        self.V = None
        self.Q = None
        self.X = None
        self.free_dofs = None
        self.gfu = None
        self.vel = None
        self.pre = None
        
        # Level set variables
        self.lset_meshadap = None
        self.deformation = None
        self.deform_last = None
        self.deform_last2 = None
        self.lsetp1 = None
        self.lset_neg = None
        self.lset_if = None
        self.ci_main = None
        
        # Extension level sets
        self.lsetp1_ext = None
        self.ci_ext = None
        
        # Element markers
        self.els = {}
        self.facets = {}
        
        # Time stepping variables
        self.t = Parameter(0.0)
        self.dt = 1 / self.dt_inv
        self.bdf1_steps = None
        self.dt_bdf1 = None
        self.gfu_last = None
        self.gfu_last_on_new_mesh = None
        self.vel_last = None
        self.gfu_last2 = None
        self.gfu_last2_on_new_mesh = None
        self.vel_last2 = None
        
        # Drag variables
        self.drag_x = 0.0
        self.drag_y = 0.0
        self.drag_x_test = None
        self.drag_y_test = None
        self.res = None
        
        # Functionals storage
        self.functionals = {"time": [], 
                            "height": [], 
                            "vel": [], 
                            "drag_x": [], 
                            "drag_y": []}
        
        # Build the problem
        self._build_mesh()
        self._build_spaces()
        self._setup_levelset()
        self._setup_forms()
        self._setup_output()
        self._setup_functionals()
        
    # ------------------------------ HELPER FUNCTIONS ------------------------------
    def _d1(self, t):
        """
        Displacement of the cylinder in the vertical direction
        """
        return 0.1 + 0.05 * cos(0.1 * pi * t)
    
    def _levelset_func(self, h):
        """
        Level set function
        """
        return (self.diam_ball / 2) - sqrt((x - self.diam_ball/3)**2 + (y - h)**2)
    
    def _v1(self, t):
        """
        Velocity of the cylinder in the z-direction
        """
        return -0.05 * 0.1 * pi * sin(0.1 * pi * t)
    
    def _circ_speed(self, t):
        """
        Velocity of the cylinder
        """
        return CoefficientFunction((0.0, self._v1(t)))
    
    def _sign(self, x):
        if x >= 0:
            return 1
        else:
            return -1
    
    def _lset_center(self, t):
        """
        Signed centre of circle
        """
        return self._sign(self._v1(t)) * (y - self._d1(t))
    
    # ------------------------------ BACKGROUND MESH ------------------------------
    def _build_mesh(self):
        """
        Build the background mesh
        """
        pnts = [self.lowerleft, (self.ur_inner[0], self.lowerleft[1]),
                (self.upperright[0], self.lowerleft[1]), self.upperright, self.ur_inner,
                (self.lowerleft[0], self.upperright[1])]
        
        # Generate geometry
        geo = SplineGeometry()
        p1, p2, p3, p4, p5, p6 = [geo.AppendPoint(*pnt) for pnt in pnts]
        
        geo.Append(["line", p1, p2], leftdomain=2, rightdomain=0, bc="wall", 
                   maxh=self.h_max / self.bottom_factor)
        geo.Append(["line", p2, p3], leftdomain=1, rightdomain=0, bc="wall")
        geo.Append(["line", p3, p4], leftdomain=1, rightdomain=0, bc="wall")
        geo.Append(["line", p4, p5], leftdomain=1, rightdomain=0, bc="slip")
        geo.Append(["line", p5, p6], leftdomain=2, rightdomain=0, bc="slip")
        geo.Append(["line", p6, p1], leftdomain=2, rightdomain=0, bc="wall")
        geo.Append(["line", p2, p5], leftdomain=2, rightdomain=1)
        
        # Generate mesh
        geo.SetDomainMaxH(1, self.h_max)
        geo.SetDomainMaxH(2, self.h_max / self.inner_factor)
        
        with TaskManager():
            self.mesh = Mesh(geo.GenerateMesh(quad_dominated=False))
        
        print(" =========== Meshing completed ========== ")
    
    # --------------------------- FINITE ELEMENT SPACE ----------------------------
    def _build_spaces(self):
        """
        Build finite element spaces
        """
        self.V = VectorH1(self.mesh, order=self.k, 
                          dirichletx="wall", dirichlety="wall|slip")
        self.Q = H1(self.mesh, order=self.k - 1)
        self.X = FESpace([self.V, self.Q], dgjumps=True)
        self.free_dofs = BitArray(self.X.ndof)
        self.gfu = GridFunction(self.X)
        self.vel, self.pre = self.gfu.components
        
        # Time stepping GridFunctions
        self.gfu_last = GridFunction(self.X)
        self.gfu_last_on_new_mesh = GridFunction(self.X)
        self.vel_last = self.gfu_last_on_new_mesh.components[0]
        self.gfu_last2 = GridFunction(self.X)
        self.gfu_last2_on_new_mesh = GridFunction(self.X)
        self.vel_last2 = self.gfu_last2_on_new_mesh.components[0]
    
    # ---------------------------- LEVELSET & CUT-INFO ----------------------------
    def _setup_levelset(self):
        """
        Setup level set and cut information
        """
        # Main level set
        self.lset_meshadap = LevelSetMeshAdaptation(
            self.mesh, order=self.k, 
            threshold=0.1, 
            discontinuous_qn=True)
        
        self.deformation = self.lset_meshadap.CalcDeformation(self._levelset_func(0.0))
        self.deform_last = GridFunction(self.deformation.space, "deform-1")
        self.deform_last2 = GridFunction(self.deformation.space, "deform-2")
        
        self.lsetp1 = self.lset_meshadap.lset_p1
        self.lset_neg = {"levelset": self.lsetp1, "domain_type": NEG, "subdivlvl": 0}
        self.lset_if = {"levelset": self.lsetp1, "domain_type": IF, "subdivlvl": 0}
        
        self.ci_main = CutInfo(self.mesh, self.lsetp1)
        
        # Extension level sets
        self.lsetp1_ext = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self._levelset_func(0.0), self.lsetp1_ext)
        self.ci_ext = CutInfo(self.mesh, self.lsetp1_ext)
        
        # ------------------------------ ELEMENT MARKERS ------------------------------
        for key in ["hasneg", "if", "ext", "middle", 
                    "active", "tmp", 
                    "act_old", "act_old2"]:
            self.els[key] = BitArray(self.mesh.ne)
            self.els[key].Clear()
        
        for key in ["active", "gp_stab", "gp_ext", "none"]:
            self.facets[key] = BitArray(self.mesh.nedge)
            self.facets[key].Clear()
    
    def _update_element_information(self, bdf, it):
        """
        Recompute element, facet and dof markers
        """
        # Physical elements
        self.ci_main.Update(self.lsetp1)
        UpdateMarkers(self.els["hasneg"], self.ci_main.GetElementsOfType(HASNEG))
        UpdateMarkers(self.els["if"], self.ci_main.GetElementsOfType(IF))
        
        # Check History
        if bdf == 1:
            CheckElementHistory(it, self.mesh.ne, self.els["hasneg"], self.els["act_old"])
        elif bdf == 2:
            CheckElementHistory(it, self.mesh.ne, self.els["hasneg"], 
                               self.els["act_old"], self.els["act_old2"])
        else:
            raise SyntaxError("Unimplemented BDF scheme requested")
        
        # Extension elements
        delta = 2 * max(abs(self._v1(self.t.Get())),
                       abs(self._v1(self.t.Get() + self.dt)),
                       abs(self._v1(self.t.Get() + 2 * self.dt))) * self.dt
        
        InterpolateToP1(self._levelset_func(self._d1(self.t)) + delta, self.lsetp1_ext)
        self.ci_ext.Update(self.lsetp1_ext)
        UpdateMarkers(self.els["ext"], self.ci_ext.GetElementsOfType(HASPOS))
        
        InterpolateToP1(self._levelset_func(self._d1(self.t.Get() + self.dt)), self.lsetp1_ext)
        self.ci_ext.Update(self.lsetp1_ext)
        UpdateMarkers(self.els["tmp"], self.ci_ext.GetElementsOfType(HASNEG))
        
        InterpolateToP1(self._levelset_func(self._d1(self.t.Get() + 2 * self.dt)), self.lsetp1_ext)
        self.ci_ext.Update(self.lsetp1_ext)
        self.els["tmp"] |= self.ci_ext.GetElementsOfType(HASNEG)
        self.els["ext"] &= self.els["tmp"]
        
        # Update active elements
        UpdateMarkers(self.els["active"], self.els["hasneg"] | self.els["ext"])
        
        # Ghost-penalty facets
        UpdateMarkers(self.facets["gp_ext"],
                     GetFacetsWithNeighborTypes(self.mesh, a=self.els["active"],
                                               b=self.els["ext"], use_and=True))
        UpdateMarkers(self.facets["gp_stab"],
                     GetFacetsWithNeighborTypes(self.mesh, a=self.els["hasneg"], 
                                               b=self.els["if"], use_and=True))
        UpdateMarkers(self.facets["active"], self.facets["gp_ext"] | self.facets["gp_stab"])
        
        # Update degrees of freedom
        UpdateMarkers(self.free_dofs,
                     CompoundBitArray([GetDofsOfElements(self.V, self.els["active"]),
                                      GetDofsOfElements(self.Q, self.els["hasneg"])]),
                     self.X.FreeDofs())
    
    # ----------------------------- (BI)LINEAR FORMS ------------------------------
    def _setup_forms(self):
        """
        Setup bilinear and linear forms
        """
        (u, p), (v, q) = self.X.TnT()
        self.v = v
        
        h = specialcf.mesh_size
        #n_mesh = specialcf.normal(self.mesh.dim)
        n_lset = 1.0 / Norm(grad(self.lsetp1)) * grad(self.lsetp1)
        self.n_lset = n_lset
        
        # Precompute parameters
        self.bdf1_steps = ceil(self.dt / (self.dt**(4 / 3)))
        self.dt_bdf1 = self.dt / self.bdf1_steps
        delta = 2 * self.dt * self.velmax
        K_tilde = int(ceil(delta / self.h_max))
        
        # Forms
        mass = self.density_fl * u * v
        stokes = self.viscosity_dyn * InnerProduct(Grad(u), Grad(v))
        stokes += - p * div(v) - q * div(u)
        stokes += - self.pReg * p * q
        
        convect = self.density_fl * InnerProduct(Grad(u) * self.vel, v)
        convect_lin = self.density_fl * InnerProduct(Grad(u) * self.vel, v)
        convect_lin += self.density_fl * InnerProduct(Grad(self.vel) * u, v)
        
        # nitsche terms
        nitsche = -self.viscosity_dyn * InnerProduct(Grad(u) * n_lset, v)
        nitsche += -self.viscosity_dyn * InnerProduct(Grad(v) * n_lset, u)
        nitsche += self.viscosity_dyn * (self.gamma_n * self.k * self.k / h) * InnerProduct(u, v)
        nitsche += p * InnerProduct(v, n_lset)
        nitsche += q * InnerProduct(u, n_lset)
        
        # ghost penalty terms
        ghost_penalty_ext = self.gamma_e * K_tilde * (self.viscosity_dyn + 1 / self.viscosity_dyn) / h**2 \
                            * (u - u.Other()) * (v - v.Other())
        
        ghost_penalty_stab = self.gamma_s * self.viscosity_dyn / h**2 * (u - u.Other()) * (v - v.Other())
        ghost_penalty_stab += -self.gamma_s / self.viscosity_dyn * (p - p.Other()) * (q - q.Other())
        
        # rhs terms
        bdf1_rhs = self.density_fl * InnerProduct(self.vel_last, v)
        
        bdf2_rhs = self.density_fl * InnerProduct(2 * self.vel_last, v)
        bdf2_rhs += -self.density_fl * InnerProduct((1 / 2) * self.vel_last2, v)
        
        nitsche_rhs = -self.viscosity_dyn * InnerProduct(Grad(v) * n_lset, self._circ_speed(self.t))
        nitsche_rhs += self.viscosity_dyn * (self.gamma_n * self.k**2 / h) * InnerProduct(self._circ_speed(self.t), v)
        nitsche_rhs += q * InnerProduct(self._circ_speed(self.t), n_lset)
        
        # Stress for drag computation
        self.stress = self.viscosity_dyn * Grad(u) * n_lset - p * n_lset
        
        # -------------------------------- INTEGRATORS --------------------------------
        def InnerBFI(form, **kwargs):
            return (SymbolicBFI(self.lset_neg, 
                               form=form.Compile(self.compile_flag, wait=self.wait_compile),
                               **kwargs), "inner")
        
        def BoundaryBFI(form, **kwargs):
            return (SymbolicBFI(self.lset_if, 
                               form=form.Compile(self.compile_flag, wait=self.wait_compile),
                               **kwargs), "boundary")
        
        def GhostPenaltyBFI(form, **kwargs):
            return (SymbolicFacetPatchBFI(form=form.Compile(self.compile_flag, wait=self.wait_compile),
                                         skeleton=False), kwargs["domain"])
        
        def InnerLFI(form, **kwargs):
            return (SymbolicLFI(self.lset_neg, 
                               form=form.Compile(self.compile_flag, wait=self.wait_compile),
                               **kwargs), "inner")
        
        def BoundaryLFI(form, **kwargs):
            return (SymbolicLFI(self.lset_if, 
                               form=form.Compile(self.compile_flag, wait=self.wait_compile),
                               **kwargs), "boundary")
        
        # Store integrator lists
        self.integrators_bdf1 = []
        self.integrators_bdf1_lin = []
        self.integrators_bdf2 = []
        self.integrators_bdf2_lin = []
        
        # BDF1
        self.integrators_bdf1.append(InnerBFI(mass + self.dt_bdf1 * (stokes + convect)))
        self.integrators_bdf1.append(BoundaryBFI(self.dt_bdf1 * nitsche))
        self.integrators_bdf1.append(GhostPenaltyBFI(self.dt_bdf1 * ghost_penalty_ext,
                                                    domain="facets_ext"))
        self.integrators_bdf1.append(GhostPenaltyBFI(self.dt_bdf1 * ghost_penalty_stab,
                                                    domain="facets_if"))
        self.integrators_bdf1.append(InnerLFI(bdf1_rhs))
        self.integrators_bdf1.append(BoundaryLFI(self.dt_bdf1 * nitsche_rhs))
        
        # BDF1 linearized
        self.integrators_bdf1_lin.append(InnerBFI(mass + self.dt_bdf1 * (stokes + convect_lin)))
        self.integrators_bdf1_lin.append(BoundaryBFI(self.dt_bdf1 * nitsche))
        self.integrators_bdf1_lin.append(GhostPenaltyBFI(self.dt_bdf1 * ghost_penalty_ext,
                                                        domain="facets_ext"))
        self.integrators_bdf1_lin.append(GhostPenaltyBFI(self.dt_bdf1 * ghost_penalty_stab,
                                                        domain="facets_if"))
        
        # BDF2
        self.integrators_bdf2.append(InnerBFI(3 / 2 * mass + self.dt * (stokes + convect)))
        self.integrators_bdf2.append(BoundaryBFI(self.dt * nitsche))
        self.integrators_bdf2.append(GhostPenaltyBFI(self.dt * ghost_penalty_ext,
                                                    domain="facets_ext"))
        self.integrators_bdf2.append(GhostPenaltyBFI(self.dt * ghost_penalty_stab,
                                                    domain="facets_if"))
        self.integrators_bdf2.append(InnerLFI(bdf2_rhs))
        self.integrators_bdf2.append(BoundaryLFI(self.dt * nitsche_rhs))
        
        # BDF2 linearized
        self.integrators_bdf2_lin.append(InnerBFI(3 / 2 * mass + self.dt * (stokes + convect_lin)))
        self.integrators_bdf2_lin.append(BoundaryBFI(self.dt * nitsche))
        self.integrators_bdf2_lin.append(GhostPenaltyBFI(self.dt * ghost_penalty_ext,
                                                        domain="facets_ext"))
        self.integrators_bdf2_lin.append(GhostPenaltyBFI(self.dt * ghost_penalty_stab,
                                                        domain="facets_if"))
        
        # Store integrator markers
        self.integrator_markers = {"inner": self.els["hasneg"],
                                   "boundary": self.els["if"],
                                   "facets_ext": self.facets["gp_ext"],
                                   "facets_if": self.facets["gp_stab"],
                                   "bottom": None}
        
        self.els_restr = {"elements": self.els["active"], "facet": self.facets["active"]}
    
    # -------------------------------- FUNCTIONALS --------------------------------
    def _setup_functionals(self):
        """
        Setup functionals for drag computation
        """
        self.drag_x_test, self.drag_y_test = GridFunction(self.X), GridFunction(self.X)
        self.drag_x_test.components[0].Set(CoefficientFunction((1.0, 0.0)))
        self.drag_y_test.components[0].Set(CoefficientFunction((0.0, 1.0)))
        self.res = self.gfu.vec.CreateVector()
    
    def _compute_drag(self):
        """
        Compute drag forces
        """
        a = RestrictedBilinearForm(self.X, element_restriction=self.els["if"],
                                   facet_restriction=self.facets["none"],
                                   check_unused=False)
        a += SymbolicBFI(self.lset_if, form=InnerProduct(self.stress, self.v),
                        definedonelements=self.els["if"])
        a.Apply(self.gfu.vec, self.res)
        self.drag_x = -InnerProduct(self.res, self.drag_x_test.vec)
        self.drag_y = -InnerProduct(self.res, self.drag_y_test.vec)
        del a
    
    # -------------------------------- OUTPUT SETUP --------------------------------
    def _setup_output(self):
        """
        Setup output files and directories
        """
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        
        # Create output file with header
        with open(self.out_dir + self.out_file, "w") as fid:
            fid.write("time\theight\tvel_ball\tdrag_r\tdrag_z\n")
        
        if self.vtk_out:
            if not os.path.isdir(self.vtk_dir):
                os.makedirs(self.vtk_dir)
            self.vtk = VTKOutput(ma=self.mesh, 
                                 coefs=[self.vel, self.pre, self.lsetp1, self.deformation],
                                names=["velocity", "pressure", "lset", "deformation"],
                                filename=self.vtk_dir + self.vtk_file, 
                                subdivision=self.vtk_subdiv)
            self.vtk.Do()
            
            # # Also output mesh
            # self.vtk_mesh = VTKOutput(ma=self.mesh, coefs=[1], names=["const"],
            #                          filename=self.vtk_dir + self.vtk_file + "Mesh",
            #                          subdivision=0)
            # self.vtk_mesh.Do()
    
    def _write_to_file(self, str_out):
        """
        Write string to output file
        """
        with open(self.out_dir + self.out_file, "a") as fid:
            fid.write(str_out)
    
    def _collect_and_write_output(self):
        """
        Collect functionals and write to file
        """
        self.functionals["time"].append(self.t.Get())
        self.functionals["height"].append(self._d1(self.t.Get()))
        self.functionals["vel"].append(self._v1(self.t.Get()))
        self.functionals["drag_y"].append(self.drag_y)
        self.functionals["drag_x"].append(self.drag_x)
        
        str_out = "{:6.4f}".format(self.functionals["time"][-1])
        for val in ["height", "vel", "drag_x", "drag_y"]:
            str_out += "\t{:9.7e}".format(self.functionals[val][-1])
        str_out += "\n"
        
        self._write_to_file(str_out)
    
    def _build_newton_system(self, integrators, integrators_lin):
        """
        Build Newton system for current time step
        """
        mStar = RestrictedBilinearForm(
            self.X, element_restriction=self.els_restr["elements"],
            facet_restriction=self.els_restr["facet"], check_unused=False)
        
        mStar_lin = RestrictedBilinearForm(
            self.X, element_restriction=self.els_restr["elements"],
            facet_restriction=self.els_restr["facet"], check_unused=False)
        
        f = LinearForm(self.X)
        
        AddIntegratorsToForm(integrators=integrators, a=mStar, f=f,
                            element_map=self.integrator_markers)
        AddIntegratorsToForm(integrators=integrators_lin, a=mStar_lin,
                            f=None, element_map=self.integrator_markers)
        
        return mStar, mStar_lin, f
    
    def _store_snapshot(self):
        """
        Store current solution as snapshot (for POD)
        """
        if not self.collect_snapshots:
            return
        
        if len(self.snapshot_times) % self.snapshot_stride == 0:
            vel_copy = GridFunction(self.V)
            vel_copy.vec.data = self.gfu.components[0].vec
            
            pre_copy = GridFunction(self.Q)
            pre_copy.vec.data = self.gfu.components[1].vec
            
            self.snapshots_velocity.append(vel_copy)
            self.snapshots_pressure.append(pre_copy)
            self.snapshot_times.append(self.t.Get())
    
    # ------------------------------- TIME STEPPING -------------------------------
    def solve(self):
        """
        Solve the problem
        """
        with TaskManager():
            # Initialize vectors
            self.gfu.vec[:] = 0
            self.gfu_last.vec[:] = 0
            self.gfu_last2.vec[:] = 0
            
            # BDF1 warm up
            print("\n=== BDF1 Warm-up ===")
            for it in range(1, self.bdf1_steps + 1):
                self.t.Set(it * self.dt_bdf1)
                
                # Store data from previous time-step
                self.gfu_last.vec.data = self.gfu.vec
                self.deform_last.vec.data = self.deformation.vec
                UpdateMarkers(self.els["act_old"], self.els["active"])
                
                self.deformation = self.lset_meshadap.CalcDeformation(
                    self._levelset_func(self._d1(self.t)))
                
                for i in range(2):
                    self.gfu_last_on_new_mesh.components[0].components[i].Set(
                        shifted_eval(self.gfu_last.components[0].components[i],
                                    self.deform_last, self.deformation))
                
                self._update_element_information(bdf=1, it=it)
                
                # Solve linearised system
                mStar, mStar_lin, f = self._build_newton_system(
                    self.integrators_bdf1, self.integrators_bdf1_lin)
                f.Assemble()
                
                CutFEM_QuasiNewton(a=mStar, alin=mStar_lin, u=self.gfu, f=f.vec,
                                  freedofs=self.free_dofs, maxit=self.maxit_newt,
                                  maxerr=self.tol_newt, inverse=self.inverse,
                                  jacobi_update_tol=self.jacobi_tol_newt, reuse=False)
                
                self._compute_drag()
                self._store_snapshot()
                
                # Do output
                self._collect_and_write_output()
                #Redraw(blocking=True)
                
                print("t = {:10.6f}, height = {:6.4f}, vel_ball = {:5.3f} active_els"
                      " = {:} - 1".format(self.t.Get(), self._d1(self.t.Get()), 
                                         self._v1(self.t.Get()), sum(self.els["active"])))
            
            # VTK output for BDF1 if needed
            if self.vtk_out and (1 * self.vtk_freq) % self.dt_inv == 0:
                projector = Projector(self.free_dofs, True)
                self.gfu.vec.data = projector * self.gfu.vec
                self.vtk.Do()
            
            # BDF2 main loop
            print("\n=== BDF2 Main Loop ===")
            self.gfu_last.vec.data = self.gfu_last2.vec
            
            for it in range(2, int(self.t_end * self.dt_inv + 0.5) + 1):
                self.t.Set(it * self.dt)
                
                # Store data from previous time-step
                self.gfu_last2.vec.data = self.gfu_last.vec
                self.gfu_last.vec.data = self.gfu.vec
                self.deform_last2.vec.data = self.deform_last.vec
                self.deform_last.vec.data = self.deformation.vec
                
                UpdateMarkers(self.els["act_old2"], self.els["act_old"])
                UpdateMarkers(self.els["act_old"], self.els["active"])
                
                self.deformation = self.lset_meshadap.CalcDeformation(
                    self._levelset_func(self._d1(self.t)))
                
                for i in range(2):
                    self.gfu_last_on_new_mesh.components[0].components[i].Set(
                        shifted_eval(self.gfu_last.components[0].components[i],
                                    self.deform_last, self.deformation))
                    self.gfu_last2_on_new_mesh.components[0].components[i].Set(
                        shifted_eval(self.gfu_last2.components[0].components[i],
                                    self.deform_last2, self.deformation))
                
                self._update_element_information(bdf=2, it=it)
                
                # Solve linearised system
                mStar, mStar_lin, f = self._build_newton_system(
                    self.integrators_bdf2, self.integrators_bdf2_lin)
                f.Assemble()
                
                CutFEM_QuasiNewton(a=mStar, alin=mStar_lin, u=self.gfu, f=f.vec,
                                  freedofs=self.free_dofs, maxit=self.maxit_newt,
                                  maxerr=self.tol_newt, inverse=self.inverse,
                                  jacobi_update_tol=self.jacobi_tol_newt, reuse=False)
                
                self._compute_drag()
                self._store_snapshot()
                
                # Do output
                self._collect_and_write_output()
                #Redraw(blocking=True)
                
                # VTK output
                if self.vtk_out and (it * self.vtk_freq) % self.dt_inv == 0:
                    projector = Projector(self.free_dofs, True)
                    self.gfu.vec.data = projector * self.gfu.vec
                    self.vtk.Do()
                
                print("t = {:10.6f}, height = {:6.4f}, vel_ball = {:5.3f} active_els"
                      " = {:}".format(self.t.Get(), self._d1(self.t.Get()), 
                                     self._v1(self.t.Get()), sum(self.els["active"])))
    
    # ------------------------------ POST-PROCESSING ------------------------------
    def print_execution_time(self):
        """
        Print total execution time
        """
        end_time = time.time() - self.start_time
        print("\n----------- Total time: {:02.0f}:{:02.0f}:{:02.0f}:{:06.3f}"
              " ----------".format(end_time // (24 * 60 * 60),
                                   end_time % (24 * 60 * 60) // (60 * 60),
                                   end_time % 3600 // 60,
                                   end_time % 60))
    
    # ------------------------------ POD-RELATED METHODS ------------------------------
    def enable_snapshot_collection(self, stride=1):
        """
        Enable collection of snapshots for POD
        Parameters:
        -----------
        stride : int
            Store every 'stride' time step
        """
        self.collect_snapshots = True
        self.snapshot_stride = stride
        self.snapshots_velocity = []
        self.snapshots_pressure = []
        self.snapshot_times = []
    
    def get_snapshots(self):
        """
        Get collected snapshots
        Returns:
        --------
        tuple: (velocity_snapshots, pressure_snapshots, snapshot_times)
        """
        return self.snapshots_velocity, self.snapshots_pressure, self.snapshot_times
    
    def save_snapshots(self, filename=None):
        """
        Save snapshots to file
        Parameters:
        -----------
        filename : str, optional
            Filename for snapshots
        """
        if filename is None:
            filename = self.out_dir + 'snapshots.pkl'
        
        # Convert GridFunctions to numpy arrays for saving
        snap_data = {
            'times': self.snapshot_times,
            'velocity': [gf.vec.FV().NumPy() for gf in self.snapshots_velocity],
            'pressure': [gf.vec.FV().NumPy() for gf in self.snapshots_pressure],
            'config': {
                'h_max': self.h_max,
                'dt_inv': self.dt_inv,
                'k': self.k,
                'density_fl': self.density_fl,
                'viscosity_dyn': self.viscosity_dyn,
                'diam_ball': self.diam_ball
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(snap_data, f)
        
        print(f"Saved {len(self.snapshots_velocity)} snapshots to {filename}")
    
    def load_snapshots(self, filename):
        """
        Load snapshots from file
        Parameters:
        -----------
        filename : str
            Filename to load from
        """
        with open(filename, 'rb') as f:
            snap_data = pickle.load(f)
        
        self.snapshot_times = snap_data['times']
        
        # Reconstruct GridFunctions from numpy arrays
        self.snapshots_velocity = []
        for vec_data in snap_data['velocity']:
            gf = GridFunction(self.V)
            gf.vec.FV().NumPy()[:] = vec_data
            self.snapshots_velocity.append(gf)
        
        self.snapshots_pressure = []
        for vec_data in snap_data['pressure']:
            gf = GridFunction(self.Q)
            gf.vec.FV().NumPy()[:] = vec_data
            self.snapshots_pressure.append(gf)
        
        print(f"Loaded {len(self.snapshots_velocity)} snapshots from {filename}")
    
    # def get_space(self):
    #     """
    #     Get the compound finite element space
    #     """
    #     return self.X
    
    # def get_component_spaces(self):
    #     """
    #     Get velocity and pressure spaces
    #     """
    #     return self.V, self.Q
