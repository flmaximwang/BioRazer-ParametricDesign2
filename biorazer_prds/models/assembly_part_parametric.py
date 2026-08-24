from abc import abstractmethod

from biorazer_prds.models.assembly_part import AssemblyPart
from biorazer_prds.models.assembly_part_ref import AssemblyPartRef

import biotite.structure as bt_struct

from dataclasses import dataclass, field


@dataclass
class AssemblyPartParaRefProperty(AssemblyPartRef):
    """
    Properties of a parametric reference part.

    For a parametric reference the ``ref_structure`` is *virtual*: it is
    generated from the fitted parameters (the ideal trace), and serves as the
    reference geometry for placement / registration.

    Properties
    ----------
    param : dict
        A dictionary to store parameters of the fitted model.
    initial_param : dict
        A dictionary to store initial parameters for fitting. You should have no
        other keys than those in params.
    extra_param : dict
        A dictionary to store extra parameters that should not be used in fitting.
    params_not_to_fit : list[str]
        Parameters that should be held fixed during fitting.
    fitted_structure : bt_struct.AtomArray
        The virtual reference structure generated from the fitted parameters.
    rmsd : float
        The root mean square deviation of the fitted model.
    """

    param: dict = field(default_factory=dict)
    initial_param: dict = field(default_factory=dict)
    extra_param: dict = field(default_factory=dict)
    params_not_to_fit: list[str] = field(default_factory=list)

    fitted_structure: bt_struct.AtomArray = None
    rmsd: float = None


@dataclass
class AssemblyPartParaRefIO(AssemblyPartParaRefProperty):
    """
    Input/Output operations for a parametric reference part.
    """

    @classmethod
    def from_params(cls, *, params: dict, **kwargs):
        """Load the structure from a given set of parameters.

        Other properties will be generated automatically based on the
        parameters.
        """
        raise NotImplementedError("from_params method is not implemented")


@dataclass
class AssemblyPartParaRefOperation(AssemblyPartParaRefProperty):
    """
    Operations on a parametric reference part.
    """

    @abstractmethod
    def fit(self, verbose: bool = False):
        """
        Fit with the given coordinates and store the parameters, rmsd and
        fitted coordinates in the object.

        self.initial_param can be used to provide initial guesses for the
        fitting.
        self.params_not_to_fit can be used to specify parameters that should
        not be fitted.
        Set verbose=True to print fitting progress information.
        """

    @abstractmethod
    def fit_with_ref(self):
        """
        Fit with the given coordinates to the reference structure and store the
        parameters, rmsd and fitted coordinates in the object.
        - Ref is mobile, the original structure is fixed.
        """

    @abstractmethod
    def modify(self, method, *args, **kwargs):
        """
        Modify the structure with the given method and arguments.
        The method should be a string that specifies the modification method.
        The args and kwargs are the arguments for the modification method.
        Modification relies on the params stored in the object.
        """


@dataclass
class AssemblyPartParaRef(
    AssemblyPartParaRefIO,
    AssemblyPartParaRefOperation,
):
    """
    A parametric reference part: its virtual ``ref_structure`` is generated
    from parameters and can be fitted to a set of coordinates.
    """
