from full_order_model import CutFEMProblem
from PodFallingBall import POD
import argparse



# -------------------------------- MAIN SCRIPT --------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CutFEM FSI with optional POD')
    parser.add_argument("--h", help="h_max", type=float, required=True)
    parser.add_argument("--dti", help="dt_inv", type=float, required=True)
    parser.add_argument("--pod", action="store_true", help="Compute POD after simulation")
    parser.add_argument("--num_modes", type=int, default=20, help="Number of POD modes")
    parser.add_argument("--snapshot_stride", type=int, default=1, help="Store every N time steps")
    parser.add_argument("--load_snapshots", type=str, help="Load snapshots from file instead of simulating")
    parser.add_argument("--load_pod", type=str, help="Load POD basis from file")
    parser.add_argument("--no_vtk", action="store_true", help="Disable VTK output")
    
    args = parser.parse_args()
    print(args)
    
    if args.pod or args.load_snapshots or args.load_pod:
        # Use POD-enabled version
        if args.no_vtk:
            # Need to modify after creation
            problem = POD(args, collect_snapshots=(args.load_snapshots is None))
            problem.vtk_out = False
        else:
            problem = POD(args, collect_snapshots=(args.load_snapshots is None))
        
        if args.load_snapshots:
            # Load existing snapshots
            problem.load_snapshots(args.load_snapshots)
            
            if args.load_pod:
                # Load existing POD basis
                problem.load_pod_basis(args.load_pod)
            else:
                # Compute POD from loaded snapshots
                problem.compute_pod(num_modes=args.num_modes)
                problem.save_pod_basis()
                problem.plot_energy_decay()
        
        elif args.pod:
            # Run full pipeline
            results = problem.run_full_pipeline(num_modes=args.num_modes)
            
            # Print final drag values
            print("\nFinal drag values:")
            if len(results['history']['drag_x']) > 0:
                print(f"  Drag X: {results['history']['drag_x'][-1]:.6e}")
                print(f"  Drag Y: {results['history']['drag_y'][-1]:.6e}")
        
        else:
            # Just simulate with snapshot collection
            problem.solve()
            problem.save_snapshots()
    
    else:
        # Use basic version (exactly like original)
        if args.no_vtk:
            problem = CutFEMProblem(args)
            problem.vtk_out = False
        else:
            problem = CutFEMProblem(args)
        
        problem.solve()
    
    problem.print_execution_time()