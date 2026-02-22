from ngsolve import Norm, Projector


def CutFEM_QuasiNewton(a, alin, u, f, freedofs, maxit=100, maxerr=1e-11, inverse="umfpack", jacobi_update_tol=0.1, reuse=False, printing=True, **kwargs):
    """
    Based on Newton() from NGSolve by J. Schöberl and NGSolve iTutorial 
    on non-linear problems by Christoph Lehrenfeld: 
    Solves the non-linear system A(u)=f using a Quasi-Newton approach 
    combined with a line-search to ensure a decreasing residual in every
    Newton step.

    Parameters
    ----------
    a : BilinearForm
      The BilinearForm of the non-linear variational problem. Note: This
      has to be assembled a the moment, therefore the non-linearity has 
      to be implicit through a GridFunction.
    alin : BilinearForm
      The linearised bilinear form of the non-linear problem.
    u : GridFunction
      The GridFunction where the solution is saved. The values are used 
      as initial guess for Newton's method.
    f : Vector
      The right-hand side vector of the non-linear variational problem. 
      Can be given as None
    freedofs : BitArray
      The FreeDofs on which the assembled matrix is inverted.
    maxit : int
      Number of maximal iteration for Newton. If the maximal number is 
      reached before the maximal error Newton might no converge and a 
      warning is displayed.
    maxerr : float
      The maximal error which Newton should reach before it stops. The 
      error is computed by the square root of the inner product of the 
      residuum and the correction.
    inverse : string
      A string of the sparse direct solver which should be solved for 
      inverting the assembled Newton matrix.
    jacobi_update_tol: float
      If the residual decreases by less than the factor 
      'jacobi_update_tol' compared to the last residual, then the 
      Jacobian is updated.
    reuse: bool
      If True, inv_jacobian is defined as a global variable and the 
      method checks in the Jacobian is available from a previous Newton 
      iteration.
    printing : bool
      Set if Newton's method should print informations about the 
      iterations like the error or if the Jacobian was updated in this 
      iteration. 
    Returns
    -------
    (int, int)
      List of two integers. The first one is 0 if Newton's method did 
      converge, -1 otherwise. The second one gives the number of Newton 
      iterations needed.
    """

    # Test for unknown keyword arguments
    for key in kwargs:
        print("WARNING: Unknown keyword argument \"{}\"".format(key))

    res = u.vec.CreateVector()
    res_freedofs = u.vec.CreateVector()
    du = u.vec.CreateVector()
    u_old = u.vec.CreateVector()
    numit = 0
    err, errLast = float("NaN"), float("NaN")
    Updated = "n/a "

    if reuse and not "inv_jacobian" in globals():
        global inv_jacobian
        JacobianAvailable = False
    if reuse and "inv_jacobian" in globals():
        JacobianAvailable = True
    else:
        JacobianAvailable = False

    projector = Projector(freedofs, True)

    if printing:
        print("    Numit\tUpd.J\t||res||_l2")
        print("    ------------------------------")

    a.Assemble()
    res.data = a.mat * u.vec
    if f:
        res.data -= f
    res_freedofs.data = projector * res
    err = Norm(res_freedofs)
    omega = 1

    for it in range(maxit):

        if printing:
            str_out = "    {:}\t\t{:}\t{:1.4e}".format(numit, Updated, err)
            if omega < 1:
                str_out += "  LS: {:}".format(omega)
            print(str_out)
        if err < maxerr:
            break
        elif not JacobianAvailable or err > errLast * jacobi_update_tol:
            UpdateJacobian = True
        else:
            UpdateJacobian = False

        numit += 1

        if UpdateJacobian:
            if JacobianAvailable:
                del inv_jacobian
            alin.Assemble()
            inv_jacobian = alin.mat.Inverse(freedofs, inverse=inverse)
            Updated = True
            JacobianAvailable = True
        else:
            Updated = False

        if alin.condense:
            res.data += alin.harmonic_extension_trans * res
            du.data = inv_jacobian * res
            du.data += alin.inner_solve * res
            du.data += alin.harmonic_extension * du
        else:
            du.data = inv_jacobian * res

        do_linesearch = True
        errLast = err
        u_old.data = u.vec
        omega = 1
        while do_linesearch and omega > 0.01:
            u.vec.data = u_old - omega * du

            a.Assemble()
            res.data = a.mat * u.vec
            if f:
                res.data -= f
            res_freedofs.data = projector * res
            err = Norm(res_freedofs)
            if err < errLast:
                do_linesearch = False
            else:
                omega *= 0.5

    else:
        print("\tWarning: Newton might not have converged: ||res|| = {:4.2e}".format(err))
        if not reuse:
            del inv_jacobian
        del res, du, res_freedofs, u_old
        
        return (-1, numit)

    if not reuse:
        try:
            del inv_jacobian
        except:
            pass
    del res, du, res_freedofs, u_old
    
    return (0, numit)
