from __future__ import absolute_import, division, print_function

import copy

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from scipy.interpolate import splev, splprep
from ndcube.extra_coords import QuantityTableCoordinate

from glue.config import viewer_tool
from glue.core import Data
from glue.core.component_link import ComponentLink
from glue.viewers.matplotlib.toolbar_mode import ToolbarModeBase

__all__ = ['LoopSelectionTool']


@viewer_tool
class LoopSelectionTool(ToolbarModeBase):
    """
    Select a single pixel, get the coordinates
    """

    icon = "glue_crosshair"
    tool_id = 'solar:loop_selection'
    action_text = 'Pixel'
    tool_tip = 'Trace a loop spine to create a derived "straightened" loop dataset'
    status_tip = 'CLICK to select points in your loop'

    _pressed = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._move_callback = self._extract_pixel
        self._press_callback = self._on_press
        self._release_callback = self._on_release
        self._derived = None

        self._selected_points, = self.viewer.axes.plot(0, 0, color='red', marker='+', markersize=10, ls='')
        self._selected_points.set_visible(False)
        self._interpolated_points, = self.viewer.axes.plot(0, 0, color='red', ls='-')
        self._interpolated_points.set_visible(False)

        # FIXME: let the user specify these via the GUI
        self._n_interp = 25
        self._loop_width = 45 * u.arcsec

    def _on_press(self, mode):
        self._pressed = True
        self._select_pixel(mode)

    def _on_release(self, mode):
        self._pressed = False

    def activate(self):
        super().activate()
        self._clicked_points = {'x': [], 'y': []}

    def deactivate(self):
        if len(self._clicked_points['x']) == 0:
            return 
        # Create coordinate dataset
        x_pix = copy.deepcopy(self._clicked_points['x'])
        y_pix = copy.deepcopy(self._clicked_points['y'])
        dc = self.viewer.session.data_collection[0]
        coord = dc.coords.pixel_to_world(x_pix, y_pix)
        d = Data(x=x_pix, y=y_pix, Tx=coord.Tx, Ty=coord.Ty, coord=coord, label='loop_points')
        self.viewer.session.data_collection.append(d)
        # Create straightened loop dataset
        coord_interp = interpolate_hpc_coord(coord, self._n_interp)
        loop_cut, xs_cut, straight_indices = straight_loop_indices(coord_interp, self._loop_width, dc.coords)
        gwcs =  (QuantityTableCoordinate(xs_cut, physical_types='pos.cartesian', names='s_perp') & 
                 QuantityTableCoordinate(loop_cut, physical_types='pos.cartesian', names='s_parallel')).wcs
        # FIXME: find a better way of selecting the image data. This is fragile! and not guaranteed to select 
        # the right component!!!
        loop_straight = dc[dc.components[-1]][straight_indices[:,:,1], straight_indices[:,:,0]]
        d_straight = Data(image=loop_straight,
                          coords=gwcs,
                          label='loop_straight')
        self.viewer.session.data_collection.append(d_straight)
        # Interpolated loop points
        d_inner = Data(x=straight_indices[:, 0, 0],
                       y=straight_indices[:, 0, 1],
                       label='loop_points_inner')
        d_outer = Data(x=straight_indices[:, -1, 0],
                       y=straight_indices[:, -1, 1],
                       label='loop_points_outer')
        i_mid = int((straight_indices.shape[1]-1)/2)
        d_mid = Data(x=straight_indices[:, i_mid, 0],
                     y=straight_indices[:, i_mid, 1],
                     label='loop_points_mid')
        self.viewer.session.data_collection.append(d_inner)
        self.viewer.session.data_collection.append(d_outer)
        self.viewer.session.data_collection.append(d_mid)
        # Create links
        links = [
            ComponentLink([dc.id['Pixel Axis 1 [x]']], d.id['x']),
            ComponentLink([dc.id['Pixel Axis 0 [y]']], d.id['y']),
            ComponentLink([dc.id['Hpln']], d.id['Tx']),
            ComponentLink([dc.id['Hplt']], d.id['Ty']),
            ComponentLink([dc.id['Pixel Axis 1 [x]']], d_inner.id['x']),
            ComponentLink([dc.id['Pixel Axis 0 [y]']], d_inner.id['y']),
            ComponentLink([dc.id['Pixel Axis 1 [x]']], d_outer.id['x']),
            ComponentLink([dc.id['Pixel Axis 0 [y]']], d_outer.id['y']),
            ComponentLink([dc.id['Pixel Axis 1 [x]']], d_mid.id['x']),
            ComponentLink([dc.id['Pixel Axis 0 [y]']], d_mid.id['y']),
        ]
        for l in links:
            self.viewer.session.data_collection.add_link(l)
        # Cleanup
        self._selected_points.set_visible(False)
        self._interpolated_points.set_visible(False)
        del self._clicked_points
        super().deactivate()

    def _select_pixel(self, mode):
        if not self._pressed:
            return
        x, y = self._event_xdata, self._event_ydata
        # TODO: check if points are already there and if so, remove them
        # TODO: check if point are already there and if so, move them
        self._clicked_points['x'].append(x)
        self._clicked_points['y'].append(y)

        self._selected_points.set_data(self._clicked_points['x'], self._clicked_points['y'])
        self._selected_points.set_visible(True)
        # Cannot do the interpolation with less than 2 points
        n_points = len(self._clicked_points['x'])
        if n_points > 1:
            # Do at most a 3rd-order spline fit
            x_interp, y_interp = self._interpolate_pixel_coords(
                self._clicked_points['x'],
                self._clicked_points['y'],
                splprep_kwargs={'k': min(n_points-1, 3)},
            )
            self._interpolated_points.set_data(x_interp, y_interp)
            self._interpolated_points.set_visible(True)
        self.viewer.axes.figure.canvas.draw()

    def _interpolate_pixel_coords(self, x, y, **kwargs):
        coord = self.viewer.session.data_collection[0].coords.pixel_to_world(x, y)
        coord_interp = interpolate_hpc_coord(coord, self._n_interp, **kwargs)
        return self.viewer.session.data_collection[0].coords.world_to_pixel(coord_interp)


def cross_section_endpoints(p0, direction, length, from_inner):
    """
    Given a point, direction, and length, find the endpoints
    of the line that is perpendicular to the line defined by
    `p0` and `direction`
    """
    angle = -np.arccos(direction[0]) * np.sign(direction[1])
    if from_inner:
        d = np.array([0, 1])
    else:
        d = np.array([-1, 1]) / 2 
    x = d * length * np.sin(angle)
    y = d * length * np.cos(angle)
    return p0[0] + x, p0[1] + y


def bresenham(x1, y1, x2, y2):
    """
    Returns an array of all pixel coordinates which the line defined by `x1, y1` and
    `x2, y2` crosses. Uses Bresenham's line algorithm to enumerate the pixels along
    a line. This was adapted from ginga

    Parameters
    ----------
    x1, y1, x2, y2 :`int`

    References
    ----------
    | https://github.com/ejeschke/ginga/blob/master/ginga/BaseImage.py#L387
    | http://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    | https://ginga.readthedocs.org/en/latest/
    """
    for x in [x1, y1, x2, y2]:
        if type(x) not in (int, np.int64):
            raise TypeError('All pixel coordinates must be of type int')
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    res = []
    x, y = x1, y1
    while True:
        res.append((x, y))
        if (x == x2) and (y == y2):
            break
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            x += sx
        if e2 <  dx:
            err = err + dx
            y += sy
    return np.array(res)


def get_inner(px_center, py_center, width):
    # For each loop segment, calculate direction
    direction = np.gradient(np.array([px_center, py_center]).astype(float), axis=1)
    direction /= np.linalg.norm(direction, axis=0)
    # Find the inner loop
    px = np.zeros(px_center.shape)
    py = np.zeros(py_center.shape)
    for i, (_px, _py) in enumerate(zip(px_center, py_center)):
        ex, ey = cross_section_endpoints([_px,_py], direction[:,i], width, False)
        px[i] = ex[0]
        py[i] = ey[0]
    return px, py


def straight_loop_indices(coord, width, image_wcs, from_inner=False):
    """
    Return pixel indices corresponding to a straightened loop defined by
    `coord` and `width`

    Parameters
    -----------
    coord
    width
    image_wcs

    Returns
    --------
    loop_cut
    xs_cut
    indices
    """
    # Get width in pixel units
    width_px,_ = image_wcs.world_to_pixel(
        SkyCoord(Tx=u.Quantity([0*width.unit, width]),
                 Ty=[0, 0]*width.unit,
                 frame=coord.frame))
    width_px = np.diff(width_px)[0]
    # Find pixels between each loop segment
    px, py = image_wcs.world_to_pixel(coord)
    if from_inner:
        px, py = get_inner(px, py, width_px)
    px = np.round(px).astype(int)
    py = np.round(py).astype(int)
    loop_pix = []
    for i in range(px.shape[0]-1):
        b = bresenham(px[i], py[i], px[i+1], py[i+1])
        # Pop the last one, unless this is the final entry because the first point
        # of the next section will be the same
        if i < px.shape[0]-2:
            b = b[:-1]
        loop_pix.append(b)

    # For each loop segment, calculate direction
    direction = np.diff(np.array([px, py]).astype(float), axis=1)
    direction /= np.linalg.norm(direction, axis=0)
    # For each pixel in each segment, find the pixels corresponding to
    # the perpendicular cut with width_px
    indices = []
    for i, seg in enumerate(loop_pix):
        _indices = []
        for p in seg:
            px, py = cross_section_endpoints(p, direction[:,i], width_px, from_inner)
            px = np.round(px).astype(int)
            py = np.round(py).astype(int)
            b = bresenham(px[0], py[0], px[-1], py[-1])
            _indices.append(b)
        indices.append(np.stack(_indices, axis=2))

    # Interpolate each perpendicular cut to make sure they have an equal number
    # of pixels
    n_xs = max([l.shape[0] for l in indices])
    if n_xs%2 == 0:
        # Always make this odd so that the "spine" of the loop corresponds to
        # a particular index
        n_xs += 1
    s = np.linspace(0, 1, n_xs)
    indices_interp = []
    for seg in indices:
        for i in range(seg.shape[-1]):
            # Parametric interpolation in pixel space 
            xs = seg[:, :, i].T
            _s = np.append(0., np.linalg.norm(np.diff(xs, axis=1), axis=0).cumsum())
            _s /= np.sum(np.diff(_s))
            tck, _ = splprep(xs, u=_s)
            px, py = splev(s, tck)
            indices_interp.append(np.array([px, py]).T)
    indices_interp = np.round(np.array(indices_interp)).astype(int)

    i_mid = int((n_xs - 1)/2)
    loop_coord = image_wcs.pixel_to_world(indices_interp[:, i_mid, 0],
                                          indices_interp[:, i_mid, 1])
    data = u.Quantity([loop_coord.Tx, loop_coord.Ty]).to('arcsec').value
    loop_cut = np.append(0., np.linalg.norm(np.diff(data, axis=1), axis=0).cumsum()) * u.arcsec
    xs_cut = width * s

    return loop_cut, xs_cut, indices_interp


def interpolate_hpc_coord(coord, n, **kwargs):
    """
    Parametric interpolation of a Helioprojective coordinate
    
    Parameters
    ----------
    coord
    n

    Returns
    -------
    coord_interp
    """
    splev_kwargs = kwargs.get('splev_kwargs', {})
    splprep_kwargs = kwargs.get('splprep_kwargs', {})
    data = u.Quantity([coord.Tx, coord.Ty]).to('arcsec').value
    s = np.append(0., np.linalg.norm(np.diff(data, axis=1), axis=0).cumsum())
    s /= np.sum(np.diff(s))
    s_new = np.linspace(0, 1, n)
    tck, _ = splprep(data, u=s, **splprep_kwargs)
    Tx, Ty = splev(s_new, tck, **splev_kwargs)
    return SkyCoord(Tx=Tx*u.arcsec, Ty=Ty*u.arcsec, frame=coord.frame)
