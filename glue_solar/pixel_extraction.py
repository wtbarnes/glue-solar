from __future__ import absolute_import, division, print_function

import copy

import numpy as np

from glue.config import viewer_tool
from glue.core import Data, Component
from glue.core.component_link import ComponentLink
from glue.core.data_derived import IndexedData
from glue.viewers.matplotlib.toolbar_mode import ToolbarModeBase

__all__ = ['PixelExtractionTool', 'PixelSelectionTool']


@viewer_tool
class PixelExtractionTool(ToolbarModeBase):
    """
    Create a derived dataset corresponding to the selected pixel.
    """

    icon = "glue_crosshair"
    tool_id = 'solar:pixel_extraction'
    action_text = 'Pixel'
    tool_tip = 'Extract data for a single pixel based on mouse location'
    status_tip = 'CLICK to select a point, CLICK and DRAG to update the extracted dataset in real time'

    _pressed = False

    def __init__(self, *args, **kwargs):
        super(PixelExtractionTool, self).__init__(*args, **kwargs)
        self._move_callback = self._extract_pixel
        self._press_callback = self._on_press
        self._release_callback = self._on_release
        self._derived = None

        self._line_x = self.viewer.axes.axvline(0, color='orange')
        self._line_x.set_visible(False)

        self._line_y = self.viewer.axes.axhline(0, color='orange')
        self._line_y.set_visible(False)

    def _on_press(self, mode):
        self._pressed = True
        self._extract_pixel(mode)

    def _on_release(self, mode):
        self._pressed = False

    def _extract_pixel(self, mode):

        if not self._pressed:
            return

        x, y = self._event_xdata, self._event_ydata

        if x is None or y is None:
            return None

        xi = int(round(x))
        yi = int(round(y))

        indices = [None] * self.viewer.state.reference_data.ndim
        indices[self.viewer.state.x_att.axis] = xi
        indices[self.viewer.state.y_att.axis] = yi

        self._line_x.set_data([x, x], [0, 1])
        self._line_x.set_visible(True)
        self._line_y.set_data([0, 1], [y, y])
        self._line_y.set_visible(True)
        self.viewer.axes.figure.canvas.draw()

        if self._derived is None:
            self._derived = IndexedData(self.viewer.state.reference_data, indices)
            self.viewer.session.data_collection.append(self._derived)
        else:
            try:
                self._derived.indices = indices
            except TypeError:
                self.viewer.session.data_collection.remove(self._derived)
                self._derived = IndexedData(self.viewer.state.reference_data, indices)
                self.viewer.session.data_collection.append(self._derived)


@viewer_tool
class PixelSelectionTool(ToolbarModeBase):
    """
    Select a single pixel, get the coordinates
    """

    icon = "glue_crosshair"
    tool_id = 'solar:pixel_selection'
    action_text = 'Pixel'
    tool_tip = 'Extract a single pixel based on mouse location'
    status_tip = 'CLICK to select points in your loop'

    _pressed = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._move_callback = self._extract_pixel
        self._press_callback = self._on_press
        self._release_callback = self._on_release
        self._derived = None

        self._selected_point, = self.viewer.axes.plot(0, 0, color='orange', marker='+', markersize=10, ls='')
        self._selected_point.set_visible(False)

    def _on_press(self, mode):
        self._pressed = True
        self._select_pixel(mode)

    def _on_release(self, mode):
        self._pressed = False

    def activate(self):
        super().activate()
        self._clicked_points = {'x': [], 'y': []}

    def deactivate(self):
        x_pix = copy.deepcopy(self._clicked_points['x'])
        y_pix = copy.deepcopy(self._clicked_points['y'])
        dc = self.viewer.session.data_collection[0]
        coord = dc.coords.pixel_to_world(x_pix, y_pix)
        d = Data(x=x_pix, y=y_pix, Tx=coord.Tx, Ty=coord.Ty, label='loop_points')
        del self._clicked_points
        self.viewer.session.data_collection.append(d)
        link_x = ComponentLink([dc.id['Pixel Axis 1 [x]']], d.id['x'])
        link_y = ComponentLink([dc.id['Pixel Axis 0 [y]']], d.id['y'])
        link_Tx = ComponentLink([dc.id['Hpln']], d.id['Tx'])
        link_Ty = ComponentLink([dc.id['Hplt']], d.id['Ty'])
        self.viewer.session.data_collection.add_link(link_x)
        self.viewer.session.data_collection.add_link(link_y)
        self.viewer.session.data_collection.add_link(link_Tx)
        self.viewer.session.data_collection.add_link(link_Ty)
        super().deactivate()

    def _select_pixel(self, mode):
        if not self._pressed:
            return
        x, y = self._event_xdata, self._event_ydata
        # TODO: check if points are already there and if so, remove them
        # TODO: check if point are already there and if so, move them
        self._clicked_points['x'].append(x)
        self._clicked_points['y'].append(y)

        self._selected_point.set_data([x,],[y,])
        self._selected_point.set_visible(True)
        self.viewer.axes.figure.canvas.draw()
