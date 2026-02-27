from full_order_model import CutFEMProblem
# ngsolve stuff
from ngsolve import *
from xfem import *
import pickle
#from pathlib import Path
import os
import time
import numpy as np


class POD(CutFEMProblem):
    """
    Extended CutFEM problem class with POD capabilities
    """
    
    def __init__(self, args=None, collect_snapshots=True, snapshot_stride=1):
        """
        Initialize with POD capabilities
        
        Parameters:
        -----------
        args : argparse.Namespace
            Command line arguments
        collect_snapshots : bool
            Whether to collect snapshots during simulation
        snapshot_stride : int
            Store every 'snapshot_stride' time step
        """
        super().__init__(args)
        
        # Enable snapshot collection
        if collect_snapshots:
            self.enable_snapshot_collection(snapshot_stride)
        
        # POD storage
        self.pod_velocity_modes = []
        self.pod_pressure_modes = []
        self.pod_velocity_energies = []
        self.pod_pressure_energies = []
        
        # Create POD directory
        self.pod_dir = self.out_dir + 'POD/'
        if not os.path.isdir(self.pod_dir):
            os.makedirs(self.pod_dir)
            
    
    def compute_inner_products(self):
        """
        Compute inner products using level set integration (most robust for CutFEM)
        """
        h = specialcf.mesh_size
        
        # Get parameters
        nu = self.viscosity_dyn   # kinematic viscosity
        rho = self.density_fl     # density
        gamma_n = self.gamma_n    # Nitsche penalty parameter
        k = self.k                # polynomial order
        
        # Velocity inner product using SymbolicBFI (handles cut elements correctly)
        u = self.V.TrialFunction()
        v = self.V.TestFunction()
        
        # Use BilinearForm instead of RestrictedBilinearForm for SymbolicBFI
        inner_u = BilinearForm(self.V)
        
        # BDF2 mass term coefficient
        inner_u += SymbolicBFI(self.lset_neg, form=u * v)
        
        # Add terms using SymbolicBFI for cut domain
        inner_u += SymbolicBFI(self.lset_neg, form=InnerProduct(Grad(u), Grad(v)))
        
        # Interface terms (Γ = zero level set)
        inner_u += SymbolicBFI(self.lset_if, form=-(Grad(u) * self.n_lset) * v)       # -(∂n u, v)_Γ
        inner_u += SymbolicBFI(self.lset_if, form=- u * (Grad(v) * self.n_lset))      # -(u, ∂n v)_Γ
        inner_u += SymbolicBFI(self.lset_if, form=gamma_n / h * u * v)                # Nitsche penalty
        
        # Ghost penalty on cut facets (if needed)
        inner_u += SymbolicBFI (self.gamma_n * self.viscosity_dyn / h**2 * (u - u.Other()) * (v - v.Other()), BND, skeleton=True)
        inner_u.Assemble()
        
        # Pressure inner product
        p = self.Q.TrialFunction()
        q = self.Q.TestFunction()
        
        # Use BilinearForm instead of RestrictedBilinearForm for SymbolicBFI
        inner_p = BilinearForm(self.Q)
        
        # L2 norm over fluid domain
        inner_p += SymbolicBFI(self.lset_neg, form=p * q)
        
        # H1 seminorm over fluid domain
        inner_p += SymbolicBFI(self.lset_neg, form=InnerProduct(Grad(p), Grad(q)))
        
        # Optional: Pressure ghost penalty for cut elements
        inner_p += SymbolicBFI(self.gamma_s / nu / h**2 * (p - p.Other()) * (q - q.Other()), BND, skeleton=True)
        
        # Adding pressure regularization (if needed)
        if hasattr(self, 'pReg'):
            inner_p += SymbolicBFI(self.lset_neg,form=self.pReg * p * q)
        
        inner_p.Assemble()
        
        return {'velocity': inner_u, 'pressure': inner_p}
    
        
    def compute_pod(self, num_modes=None):
        """
        Compute POD from collected snapshots
        Parameters:
        -----------
        num_modes : int, optional
            Number of modes to retain  
        Returns:
        --------
        dict: POD modes and energies
        """
        if len(self.snapshots_velocity) == 0:
            raise ValueError("No snapshots available. Run solve() first or load snapshots.")
        
        if num_modes is None:
            num_modes = min(20, len(self.snapshots_velocity))
        
        # Get inner products
        inner_prods = self.compute_inner_products()
        
        # Compute POD for velocity
        print("\nComputing velocity POD...")
        self.pod_velocity_modes, self.pod_velocity_energies = self._snapshot_pod(
            self.snapshots_velocity, inner_prods['velocity'], num_modes
        )
        
        # Compute POD for pressure
        print("Computing pressure POD...")
        self.pod_pressure_modes, self.pod_pressure_energies = self._snapshot_pod(
            self.snapshots_pressure, inner_prods['pressure'], num_modes
        )
        
        # Print energy capture
        cum_energy_v = np.cumsum(self.pod_velocity_energies) / np.sum(self.pod_velocity_energies)
        cum_energy_p = np.cumsum(self.pod_pressure_energies) / np.sum(self.pod_pressure_energies)
        
        print(f"\nVelocity POD: {len(self.pod_velocity_modes)} modes")
        if len(cum_energy_v) > 4:
            print(f"  Energy capture (5 modes): {cum_energy_v[4]*100:.2f}%")
        if len(cum_energy_v) > 9:
            print(f"  Energy capture (10 modes): {cum_energy_v[9]*100:.2f}%")
        
        print(f"Pressure POD: {len(self.pod_pressure_modes)} modes")
        if len(cum_energy_p) > 4:
            print(f"  Energy capture (5 modes): {cum_energy_p[4]*100:.2f}%")
        if len(cum_energy_p) > 9:
            print(f"  Energy capture (10 modes): {cum_energy_p[9]*100:.2f}%")
        
        return {
            'velocity': {
                'modes': self.pod_velocity_modes,
                'energies': self.pod_velocity_energies
            },
            'pressure': {
                'modes': self.pod_pressure_modes,
                'energies': self.pod_pressure_energies
            }
        }
    
    def _snapshot_pod(self, snapshots, inner_product, num_modes):
        """
        Snapshot POD method
        Parameters:
        -----------
        snapshots : list
            List of GridFunctions
        inner_product : BilinearForm
            Inner product matrix
        num_modes : int
            Number of modes to retain   
        Returns:
        --------
        tuple: (modes, eigenvalues)
        """
        n_snap = len(snapshots)
        
        print(f"  Building correlation matrix ({n_snap} x {n_snap})...")
        # Build correlation matrix
        C = np.zeros((n_snap, n_snap))
        
        for i in range(n_snap):
            vi = snapshots[i].vec
            for j in range(i, n_snap):
                vj = snapshots[j].vec
                
                if hasattr(inner_product, 'mat'):
                    val = InnerProduct(vi, inner_product.mat * vj)
                    #print(f"  Inner product with mat: C[{i}, {j}] = {val:.2e}")
                else:
                    val = InnerProduct(vi, vj)
                    #print(f"  Standard inner product: C[{i}, {j}] = {val:.2e}")
                
                C[i, j] = val
                C[j, i] = val
        
        # Solve eigenvalue problem
        print("  Solving eigenvalue problem...")
        eigenvals, eigenvecs = np.linalg.eigh(C)
        
        # Sort by descending eigenvalue
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Compute modes
        print(f"  Computing {min(num_modes, n_snap)} modes...")
        modes = []
        
        for i in range(min(num_modes, n_snap)):
            if eigenvals[i] < 1e-12:
                break
            
            # Create mode as linear combination of snapshots
            mode = GridFunction(snapshots[0].space)
            mode.vec[:] = 0
            
            for j in range(n_snap):
                mode.vec.data += eigenvecs[j, i] * snapshots[j].vec
            
            # Normalize
            if hasattr(inner_product, 'mat'):
                norm = sqrt(InnerProduct(mode.vec, inner_product.mat * mode.vec))
                #print(f"  Mode {i}: eigenvalue={eigenvals[i]:.2e}, norm={norm:.2e} (with inner product mat)")
            else:
                norm = sqrt(InnerProduct(mode.vec, mode.vec))
                #print(f"  Mode {i}: eigenvalue={eigenvals[i]:.2e}, norm={norm:.2e} (standard inner product)")
            
            if norm > 0:
                mode.vec.data /= norm
                modes.append(mode)
        
        print(f"  Computed {len(modes)} modes")
        
        return modes, eigenvals[:len(modes)]
    
    def save_pod_basis(self, filename=None):
        """
        Save POD basis to file
        Parameters:
        -----------
        filename : str, optional
            Filename for POD basis
        """
        if filename is None:
            filename = self.pod_dir + 'pod_basis.pkl'
        
        # Convert modes to numpy arrays for saving
        pod_data = {
            'velocity_modes': [mode.vec.FV().NumPy() for mode in self.pod_velocity_modes],
            'pressure_modes': [mode.vec.FV().NumPy() for mode in self.pod_pressure_modes],
            'velocity_energies': self.pod_velocity_energies,
            'pressure_energies': self.pod_pressure_energies,
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
            pickle.dump(pod_data, f)
        
        print(f"Saved POD basis to {filename}")
        
        # Also save VTK visualization of modes
        if len(self.pod_velocity_modes) > 0:
            vtk_dir = self.pod_dir + 'vtk_modes/'
            if not os.path.isdir(vtk_dir):
                os.makedirs(vtk_dir)
            
            # Save first few velocity modes
            n_modes_v = min(5, len(self.pod_velocity_modes))
            if n_modes_v > 0:
                vtk_v = VTKOutput(
                    ma=self.mesh,
                    coefs=self.pod_velocity_modes[:n_modes_v],
                    names=[f"vel_mode_{i}" for i in range(n_modes_v)],
                    filename=vtk_dir + 'velocity_modes',
                    subdivision=self.vtk_subdiv
                )
                vtk_v.Do()
            
            # Save first few pressure modes
            n_modes_p = min(5, len(self.pod_pressure_modes))
            if n_modes_p > 0:
                vtk_p = VTKOutput(
                    ma=self.mesh,
                    coefs=self.pod_pressure_modes[:n_modes_p],
                    names=[f"pre_mode_{i}" for i in range(n_modes_p)],
                    filename=vtk_dir + 'pressure_modes',
                    subdivision=self.vtk_subdiv
                )
                vtk_p.Do()
    
    def load_pod_basis(self, filename):
        """
        Load POD basis from file
        
        Parameters:
        -----------
        filename : str
            Filename to load from
        """
        with open(filename, 'rb') as f:
            pod_data = pickle.load(f)
        
        # Reconstruct GridFunctions from numpy arrays
        self.pod_velocity_modes = []
        for vec_data in pod_data['velocity_modes']:
            mode = GridFunction(self.V)
            mode.vec.FV().NumPy()[:] = vec_data
            self.pod_velocity_modes.append(mode)
        
        self.pod_pressure_modes = []
        for vec_data in pod_data['pressure_modes']:
            mode = GridFunction(self.Q)
            mode.vec.FV().NumPy()[:] = vec_data
            self.pod_pressure_modes.append(mode)
        
        self.pod_velocity_energies = pod_data['velocity_energies']
        self.pod_pressure_energies = pod_data['pressure_energies']
        
        print(f"Loaded POD basis with {len(self.pod_velocity_modes)} velocity modes "
            f"and {len(self.pod_pressure_modes)} pressure modes")
    
    def plot_energy_decay(self, save=True, show=False):
        """
        Plot energy decay of POD modes
        Parameters:
        -----------
        save : bool
            Save figure to file
        show : bool
            Show figure interactively
        """
        try:
            import matplotlib.pyplot as plt
            
            if len(self.pod_velocity_energies) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                cum_energy_v = np.cumsum(self.pod_velocity_energies) / np.sum(self.pod_velocity_energies)
                
                axes[0].semilogy(self.pod_velocity_energies / self.pod_velocity_energies[0], 
                                'o-', label='Velocity')
                axes[0].set_xlabel('Mode index')
                axes[0].set_ylabel('Normalized eigenvalue')
                axes[0].set_title('Velocity POD Energy Decay')
                axes[0].grid(True)
                axes[0].legend()
                
                axes[1].plot(cum_energy_v * 100, 's-', label='Velocity')
                axes[1].set_xlabel('Number of modes')
                axes[1].set_ylabel('Cumulative energy (%)')
                axes[1].set_title('Velocity Energy Capture')
                axes[1].grid(True)
                axes[1].axhline(y=99, color='r', linestyle='--', label='99%')
                axes[1].legend()
                
                plt.tight_layout()
                if save:
                    fig.savefig(self.pod_dir + 'velocity_pod_energy.png', dpi=150)
                if show:
                    plt.show()
                else:
                    plt.close()
            
            if len(self.pod_pressure_energies) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                cum_energy_p = np.cumsum(self.pod_pressure_energies) / np.sum(self.pod_pressure_energies)
                
                axes[0].semilogy(self.pod_pressure_energies / self.pod_pressure_energies[0], 
                                'o-', label='Pressure')
                axes[0].set_xlabel('Mode index')
                axes[0].set_ylabel('Normalized eigenvalue')
                axes[0].set_title('Pressure POD Energy Decay')
                axes[0].grid(True)
                axes[0].legend()
                
                axes[1].plot(cum_energy_p * 100, 's-', label='Pressure')
                axes[1].set_xlabel('Number of modes')
                axes[1].set_ylabel('Cumulative energy (%)')
                axes[1].set_title('Pressure Energy Capture')
                axes[1].grid(True)
                axes[1].axhline(y=99, color='r', linestyle='--', label='99%')
                axes[1].legend()
                
                plt.tight_layout()
                if save:
                    fig.savefig(self.pod_dir + 'pressure_pod_energy.png', dpi=150)
                if show:
                    plt.show()
                else:
                    plt.close()
                    
        except ImportError:
            print("matplotlib not available for plotting")
    
    def run_full_pipeline(self, num_modes=20):
        """
        Run complete pipeline: FOM simulation + POD
        Parameters:
        -----------
        num_modes : int
            Number of POD modes to compute
        Returns:
        --------
        dict: Results including history and POD modes
        """
        print("="*60)
        print("Starting Full-Order Model Simulation with POD")
        print("="*60)
        
        # Run FOM simulation
        start_time = time.time()
        self.solve()
        fom_time = time.time() - start_time
        
        print(f"\nFOM simulation completed in {fom_time:.2f} seconds")
        print(f"Collected {len(self.snapshots_velocity)} snapshots")
        
        # Compute POD
        start_time = time.time()
        pod_results = self.compute_pod(num_modes=num_modes)
        pod_time = time.time() - start_time
        
        print(f"\nPOD computation completed in {pod_time:.2f} seconds")
        
        # Save results
        self.save_snapshots()
        self.save_pod_basis()
        self.plot_energy_decay()
        
        # Summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETED - SUMMARY")
        print("="*60)
        print(f"Time steps: {len(self.functionals['time'])}")
        print(f"Snapshots: {len(self.snapshots_velocity)}")
        print(f"Velocity modes: {len(self.pod_velocity_modes)}")
        print(f"Pressure modes: {len(self.pod_pressure_modes)}")
        print("="*60)
        
        return {
            'history': self.functionals,
            'pod': pod_results,
            'timing': {'fom': fom_time, 'pod': pod_time}
        }


