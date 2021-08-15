from pkg_resources import get_distribution, DistributionNotFound
from sunpy.visualization.colormaps import cmlist
from glue.viewers.image.qt import ImageViewer
from glue_solar.pixel_extraction import *  # noqa
from glue_solar.loop_selection import *  # noqa
from glue.config import colormaps
#from glue_solar.instruments import *
from glue_solar.core import *

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass


def setup():
    ImageViewer.tools.append('solar:pixel_extraction')
    ImageViewer.tools.append('solar:loop_selection')
    for _, ctable in sorted(cmlist.items()):
        colormaps.add(ctable.name, ctable)
