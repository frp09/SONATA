#------------------------------------------------------------------------------------
# Import Libraries
#------------------------------------------------------------------------------------

# Core Library modules
import numbers
import os

# Third party modules
import numpy as np
# PythonOCC Libraries
from OCC.Core.gp import gp_Pnt

# First party modules
from SONATA.cbm.topo.BSplineLst_utils import (BSplineLst_from_dct,)
from SONATA.cbm.topo.wire_utils import (build_wire_from_BSplineLst,
                                        trsf_wire,)
from SONATA.utl.trsf import trsf_af_to_blfr

# SONATA modules:
if __name__ == "__main__":
    os.chdir("..")



class Airfoil(object):
    """
    Airfoil Class object.

    Attributes
    ----------
    name : string
        The name of the airfoil

    coordinates : np.array
        The x and y coordinates of the airfoil as np.array of shape (...,2)
        x coordinates should be defined between 0 and 1.

    polars: list
        A list of Polar instances. That store c_l, c_d, and c_m together with
        Reynolds Number re and Machnumber ma.

    relative thickness : float
        Relative Thickness of the airfoil

    Notes
    ----------
    - A very brief parameter check is only performed at __init__ and is not
    implemented via a getter and setter functionality to save function calls
    and to preserve simplicity

    """

    class_counter = 1  # class attribute
    __slots__ = ("name", "id", "coordinates", "polars", "relative_thickness", "wire", "BSplineLst", "display", "start_display", "add_menu", "add_function_to_menu")

    def __init__(self, yml=None, name="NONAME", coordinates=None, polars=None, relative_thickness=None):
        self.name = "NONAME"
        self.id = self.__class__.class_counter
        self.__class__.class_counter += 1

        self.coordinates = None
        self.polars = None
        self.relative_thickness = None
        self.wire = None
        self.BSplineLst = None

        if isinstance(yml, dict):
            self.read_yaml_airfoil(yml)

        if isinstance(name, str) and not "NONAME":
            self.name = name

        if isinstance(coordinates, np.ndarray) and coordinates.shape[1] == 2:
            self.coordinates = coordinates

        if isinstance(relative_thickness, numbers.Real) and relative_thickness >= 0:
            self.relative_thickness = relative_thickness

    def __repr__(self):
        """__repr__ is the built-in function used to compute the "official"
        string reputation of an object, """
        return "Airfoil: " + self.name

    @property
    def te_coordinates(self):
        """
        returns the calculated trailing edge coordinates. Mean of the first and
        the last coordinate point
        """
        te = np.mean(np.vstack((self.coordinates[0], self.coordinates[-1])), axis=0)
        return te

    def read_yaml_airfoil(self, yml):
        """
        reads the Airfoil dictionary from the yaml dictionary and assigns them to
        the class attributes
        """
        self.name = yml["name"]
        self.relative_thickness = yml.get("relative_thickness")

        self.coordinates = np.asarray([yml["coordinates"]["x"], yml["coordinates"]["y"]], dtype=float).T
        # Shifting coordinates such that the y values of the first and last coordinate average to 0
        yshift = (self.coordinates[0][1]+self.coordinates[-1][1])/2
        for i in range(self.coordinates.shape[0]):
            self.coordinates[i][1] = self.coordinates[i][1]-yshift
        print(" ")



    def gen_OCCtopo(self, angular_deflection = 30 ):
        """
        generates a Opencascade TopoDS_Wire and BSplineLst from the airfoil coordinates.
        This can be used for interpolation and surface generation


        Returns
        ---------
        self.wire : TopoDS_Wire
            wire of the airfoil

        """
        data = np.hstack((self.coordinates, np.zeros((self.coordinates.shape[0], 1))))
        self.BSplineLst = BSplineLst_from_dct(data, angular_deflection=angular_deflection, closed=True, tol_interp=1e-5, twoD=False)
        # print("STATUS:\t CHECK Head2Tail: \t\t ", Check_BSplineLst_Head2Tail(self.BSplineLst))
        # print("STATUS:\t CHECK Counterclockwise: \t ", BSplineLst_Orientation(self.BSplineLst, 11))

        self.wire = build_wire_from_BSplineLst(self.BSplineLst, twoD=False)
        return self.wire

    def trsf_to_blfr(self, loc, soy, chord, twist):
        """
        transforms the nondim. airfoil to the blade reference frame location
        and pitch-axis information, scales it with chord information and rotates
        it with twist information

        Parameters
        ----------
        loc : array
            [x,y,z] position in blade reference coordinates
        soy : float
            dim. pitch axis location
        chord : float
            chordlength
        twist : float
            twist angle about x in radians

        Retruns
        ---------
        wire : TopoDS_Wire

        """
        Trsf = trsf_af_to_blfr(loc, soy, chord, twist)
        if self.wire is None or self.BSplineLst is None:
            self.gen_OCCtopo()

        wire = trsf_wire(self.wire, Trsf)
        tmp_pnt = gp_Pnt(self.te_coordinates[0], self.te_coordinates[1], 0)
        te_pnt = tmp_pnt.Transformed(Trsf)
        return (wire, te_pnt)  # bspline, nodes, normals

    def check_flatback(self, coords):
        count = 0
        for x in coords[:,0]:
            if x > 0.9:
                count +=1
        if count > 18:
            print("\n\n\nFLATBACK\n\n\n")
            return True
        else:
            print("\n\n\nNOT A FLATBACK\n\n\n")
            return False

    def normalize_points(self, shape, num_points):
        """Normalize the number of points in the shape to num_points."""
        x = shape[:, 0]
        y = shape[:, 1]

        # Parameterize the original shape
        t = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, num_points)

        # Interpolate x and y coordinates
        x_new = np.interp(t_new, t, x)
        y_new = np.interp(t_new, t, y)

        return np.vstack((x_new, y_new)).T

    def interpolate_shapes(self, af1, af2, t):
        """Interpolate between shape1 and shape2 based on parameter t (0 <= t <= 1)."""
        # Ensure both shapes have the same number of points
        if np.mean(af1[int(len(af1)*1/6):int(len(af1)*1/3), 1]) < 0:
            af1 = np.flip(af1,0)
        if np.mean(af2[int(len(af2)*1/6):int(len(af2)*1/3), 1]) < 0:
            af2 = np.flip(af2,0)
        num_points = max(len(af1), len(af2))
        af1 = self.normalize_points(af1, num_points)
        af2 = self.normalize_points(af2, num_points)

        # Interpolate between shapes
        interpolated_shape = (1 - t) * af1 + t * af2

        # Adjust to maintain the property that the first and last y values average to 0
        avg_y = (interpolated_shape[0, 1] + interpolated_shape[-1, 1]) / 2
        interpolated_shape[0, 1] -= avg_y
        interpolated_shape[-1, 1] -= avg_y

        return interpolated_shape

    def _scale_coords_by_chord(self, coords, chord):
        """
        Scale airfoil coordinates by a dimensional chord value.

        Converts nondimensional coordinates to dimensional by multiplying by chord.

        Parameters
        ----------
        coords : ndarray
            (N, 2) array of nondimensional coordinates [x, y]
        chord : float
            Dimensional chord length to scale by

        Returns
        -------
        ndarray
            (N, 2) array of dimensional coordinates
        """
        return coords * chord

    def _unscale_coords_by_chord(self, coords, chord):
        """
        Unscale airfoil coordinates by a dimensional chord value.

        Converts dimensional coordinates back to nondimensional by dividing by chord.

        Parameters
        ----------
        coords : ndarray
            (N, 2) array of dimensional coordinates [x, y]
        chord : float
            Dimensional chord length to divide by

        Returns
        -------
        ndarray
            (N, 2) array of nondimensional coordinates
        """
        return coords / chord

    def _shift_coords_to_pa_origin(self, coords, pa_nondim):
        """
        Shift airfoil coordinates to pitch-axis-origin reference frame.

        Translates x-coordinates so that pitch axis is at x=0 instead of leading edge.
        Useful for performing morphing in physically meaningful reference frame.

        Parameters
        ----------
        coords : ndarray
            (N, 2) array of LE-origin nondimensional coordinates [x, y]
        pa_nondim : float
            Pitch axis position (nondimensional fraction of chord from LE)

        Returns
        -------
        ndarray
            (N, 2) array with origin shifted to PA (x_new = x - pa_nondim)
        """
        coords_shifted = coords.copy()
        coords_shifted[:, 0] -= pa_nondim
        return coords_shifted

    def _shift_coords_from_pa_origin(self, coords, shift_amount):
        """
        Shift coordinates back from pitch-axis-origin to LE-origin reference frame.

        Undoes the PA-origin shift by translating back to normal LE-origin coordinates.

        Parameters
        ----------
        coords : ndarray
            (N, 2) array of PA-origin coordinates [x, y]
        shift_amount : float
            Amount to shift (typically min_x of interpolated airfoil)

        Returns
        -------
        ndarray
            (N, 2) array shifted back to LE-origin (x_new = x + shift_amount)
        """
        coords_shifted = coords.copy()
        coords_shifted[:, 0] += shift_amount
        return coords_shifted

    def interpolate_chord_scaled(self, airfoil2, k=0.5, chord1=None, chord2=None, pa1=None, pa2=None):
        """
        Performs linear interpolation of two airfoils accounting for chord scaling and pitch axis.

        This method scales airfoils to their dimensional chords before interpolation,
        optionally shifting to pitch-axis-origin frame for more physically accurate morphing.
        The result is returned in nondimensional LE-origin coordinates.

        Parameters
        ----------
        airfoil2 : Airfoil
            The airfoil to interpolate toward
        k : float, optional
            Interpolation parameter (0 = self, 1 = airfoil2). Default is 0.5.
        chord1 : float, optional
            Dimensional chord of first airfoil. If None, falls back to standard transformation.
        chord2 : float, optional
            Dimensional chord of second airfoil. If None, falls back to standard transformation.
        pa1 : float, optional
            Pitch axis position (nondimensional) of first airfoil. If provided with pa2,
            enables pitch-axis-aware interpolation.
        pa2 : float, optional
            Pitch axis position (nondimensional) of second airfoil. If provided with pa1,
            enables pitch-axis-aware interpolation.

        Returns
        -------
        Airfoil
            Interpolated airfoil in nondimensional LE-origin frame with name
            "INTERP_<name1>_<name2>_<k>"
        """
        # Fallback to standard transformation if no enhancements provided
        if (chord1 is None or chord2 is None) and (pa1 is None or pa2 is None):
            return self.transformed(airfoil2, k)

        # Start with coordinates in LE-origin frame
        coords1 = self.coordinates
        coords2 = airfoil2.coordinates

        # Step 1: If PA values provided, shift to PA-origin frame first
        if pa1 is not None and pa2 is not None:
            coords1 = self._shift_coords_to_pa_origin(coords1, pa1)
            coords2 = self._shift_coords_to_pa_origin(coords2, pa2)

        # Step 2: Scale by chord (if provided)
        if chord1 is not None and chord2 is not None:
            coords1_scaled = self._scale_coords_by_chord(coords1, chord1)
            coords2_scaled = self._scale_coords_by_chord(coords2, chord2)
        else:
            coords1_scaled = coords1
            coords2_scaled = coords2

        # Step 3: Create temporary airfoils and interpolate in scaled space
        af1_temp = Airfoil(coordinates=coords1_scaled)
        af2_temp = Airfoil(coordinates=coords2_scaled)
        af_interp_scaled = af1_temp.transformed(af2_temp, k)

        # Step 4: Compute result chord from interpolated airfoil geometry (max_x - min_x)
        interp_coords = af_interp_scaled.coordinates
        computed_chord = np.max(interp_coords[:, 0]) - np.min(interp_coords[:, 0])

        # Step 5: Unscale result back to nondimensional
        coords_result = self._unscale_coords_by_chord(af_interp_scaled.coordinates, computed_chord)

        # Step 6: If PA shift was applied, shift result back to LE-origin using min_x
        if pa1 is not None and pa2 is not None:
            min_x = np.min(coords_result[:, 0])
            coords_result[:, 0] -= min_x  # Shift so LE is at x=0

        # Step 7: Return result airfoil
        result = Airfoil(coordinates=coords_result)
        str_k = "%.3f" % k
        result.name = "INTERP_" + self.name + "_" + airfoil2.name + "_" + str_k.replace(".", "")

        return result

    def transformed(self, airfoil2, k=0.5):
        """
        Performs and linear interpolation of the airfoil with another airfoil
        by adding points to the airfoil with less coordinate pairs until the
        airfoils have the same number of points. A condition is then applied
        forcing the first and last y values of the interpolated coordinates
        to average to 0.

        Parameters
        ----------
        airfoil2 : Airfoil
            The airfoil the user whats the current airfoil to be transformed to
        k : float
            the vectors magnitude factor. k=0: the transformed airfoil remains
            the airfoil. k=1: the transformed airfoil will become airfoil2

        Returns:
        ----------
        trf_af : Airfoil
            with the name = TRF_airfoil1_airfoil2_k

        """

        trf_af = Airfoil()
        str_k = "%.3f" % k
        trf_af.name = "TRF_" + self.name + "_" + airfoil2.name + "_" + str_k.replace(".", "")
        trf_af.coordinates = self.interpolate_shapes(self.coordinates, airfoil2.coordinates, k)

        return trf_af
