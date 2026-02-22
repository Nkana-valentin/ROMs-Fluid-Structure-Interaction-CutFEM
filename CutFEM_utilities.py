from itertools import chain
from ngsolve import COUPLING_TYPE, BFI, LFI, BitArray
from xfem import GetElementsWithNeighborFacets


def UpdateMarkers(element_marker, union_elememts, intersection_elements=None):
    """
    Updated marker BitArray element_markers from one union_elements. If intersection_elements
    is given, it updates element_markers as the intersection of the last two BitArrays

    Parameters
    ----------
    element_marker : ngsolve.ngstd.BitArray
            BitArray to be updated.
    union_elememts : ngsolve.ngstd.BitArray
            BitArray to always written into the element_marker BitArray
    intersection_elements : ngsolve.ngstd.BitArray
            If given, the intersection of intersection_elements and union_elememts will be returned

    Returns
    -------
            None
    """
    element_marker.Clear()
    element_marker |= union_elememts
    if intersection_elements:
        element_marker &= intersection_elements

    return None


def CheckElementHistory(it, mesh_els, els_hasneg_current, els_extention_old, els_extention_old2=None, els_extention_old3=None, **kwargs):
    """
    Checks if all current elements have the necessary history for the BDF1, BDF2
    or BDF3 method, assuming that the initial u0 is sufficiently extended.
    Overloaded Function

    1. CheckElementHistory(int, ngsolve.ngstd.BitArray, ngsolve.ngstd.BitArray)
            Checks history for BDF1 

    2. CheckElementHistory(int, ngsolve.ngstd.BitArray, ngsolve.ngstd.BitArray, ngsolve.ngstd.BitArray)
            Checks history for BDF2 if it > 2. 

    3. CheckElementHistory(int, ngsolve.ngstd.BitArray, ngsolve.ngstd.BitArray, ngsolve.ngstd.BitArray, ngsolve.ngstd.BitArray)
            Checks history for BDF3 if it > 3. For smaller it, checks the appropriate BDFx history.

    Parameters
    ----------
    it : int
            The iteration number of the BDF scheme
    mesh_els : int
            Number of elements in the mesh
    els_hasneg_current : ngsolve.ngstd.BitArray
            Elements which require a history
    els_extention_old  : ngsolve.ngstd.BitArray
            Extension elements at step it-1. Always required.
    els_extention_old2 : ngsolve.ngstd.BitArray
            Extension elements at step it-1. Required for BDF2 and BDF3.
    els_extention_old3 : ngsolve.ngstd.BitArray
            Extension elements at step it-1. Required for BDF3 only.

    Returns
    -------
    None

    Raises
    ------
    Exception
            If some elements do not have the required history for the BDF scheme used.
    """

    for key in kwargs:
        print("WARNING: Unknown Keyword argument \"{}\"".format(key))

    if it <= 1:
        return None
    else:
        els_test, els_intersection = BitArray(mesh_els), BitArray(mesh_els)
        els_test.Clear(), els_intersection.Clear()

        els_intersection |= els_extention_old
        if it > 2 and els_extention_old2 is not None:
            els_intersection &= els_extention_old2

            if it > 3 and els_extention_old3 is not None:
                els_intersection &= els_extention_old3

        els_test |= els_hasneg_current
        els_test &= ~els_intersection
        
        if sum(els_test) > 0:
            raise Exception("Some elements have no history !!!")
        else:
            return None


def AddIntegratorsToForm(integrators, a, f, element_map, **kwargs):
    """
    Adds integrators (SymbolicBFI/SymbolicFacetPatchBFI/SymbolicLFI) to the 
    bilinear form "a" and linear form "f" and sets the respective definedonelements 
    for each integrator.

    Parameters
    ----------
    integrators : list
            List containing tuples of integrator and domain region for integrator.
            Valid domains are: "inner","boundary","facetpatch".
    a : BilinearForm
            The form into which BFI integrators will be added.
    f : LinearForm
            The form into which LFI integrators will be added. Can be None
    element_map : dictionary
            Map the domain string to bit array

    Returns
    -------
    None
    """
    for key in kwargs:
        print("WARNING: Unknown Keyword argument \"{}\"".format(key))

    for integrator, domain in integrators:
        try:
            integrator.SetDefinedOnElements(element_map[domain])
        except KeyError:
            raise TypeError("integrator domain unknown")

        if type(integrator) == BFI:
            a += integrator
        elif f is not None and type(integrator) == LFI:
            f += integrator
        else:
            raise TypeError("Integrator type unknown")


