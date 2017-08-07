# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Miscellaneous phantoms that do not fit in other categories."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import sys


__all__ = ('submarine', 'text', 'disc_phantom', 'donut', 'sphere', 'sphere2', 'cube')


def submarine(space, smooth=True, taper=20.0):
    """Return a 'submarine' phantom consisting in an ellipsoid and a box.

    Parameters
    ----------
    space : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created.
    smooth : bool, optional
        If ``True``, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : float, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : ``space`` element
        The submarine phantom in ``space``.
    """
    if space.ndim == 2:
        if smooth:
            return _submarine_2d_smooth(space, taper)
        else:
            return _submarine_2d_nonsmooth(space)
    else:
        raise ValueError('phantom only defined in 2 dimensions, got {}'
                         ''.format(space.ndim))


def _submarine_2d_smooth(space, taper):
    """Return a 2d smooth 'submarine' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_ellipse(x):
        """Blurred characteristic function of an ellipse.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the ellipse is centered at ``(0.6, 0.3)`` and has half-axes
        ``(0.4, 0.14)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.4, 0.14]) * space.domain.extent()
        center = np.array([0.6, 0.3]) * space.domain.extent()
        center += space.domain.min()

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    def blurred_rect(x):
        """Blurred characteristic function of a rectangle.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the rect has lower left ``(0.56, 0.4)`` and upper right
        ``(0.76, 0.6)``. For other domains, the values are scaled
        accordingly.
        """
        xlower = np.array([0.56, 0.4]) * space.domain.extent()
        xlower += space.domain.min()
        xupper = np.array([0.76, 0.6]) * space.domain.extent()
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            length = upp - low
            out = out * (logistic((xi - low) / length, taper) *
                         logistic((upp - xi) / length, taper))
        return out

    out = space.element(blurred_ellipse)
    out += space.element(blurred_rect)
    return out.ufuncs.minimum(1, out=out)


def _submarine_2d_nonsmooth(space):
    """Return a 2d nonsmooth 'submarine' phantom."""

    def ellipse(x):
        """Characteristic function of an ellipse.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the ellipse is centered at ``(0.6, 0.3)`` and has half-axes
        ``(0.4, 0.14)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.4, 0.14]) * space.domain.extent()
        center = np.array([0.6, 0.3]) * space.domain.extent()
        center += space.domain.min()

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    def rect(x):
        """Characteristic function of a rectangle.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the rect has lower left ``(0.56, 0.4)`` and upper right
        ``(0.76, 0.6)``. For other domains, the values are scaled
        accordingly.
        """
        xlower = np.array([0.56, 0.4]) * space.domain.extent()
        xlower += space.domain.min()
        xupper = np.array([0.76, 0.6]) * space.domain.extent()
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            out = out * ((xi >= low) & (xi <= upp))
        return out

    out = space.element(ellipse)
    out += space.element(rect)
    return out.ufuncs.minimum(1, out=out)


def text(space, text, font=None, border=0.2, inverted=True):
    """Create phantom from text.

    The text is represented by a scalar image taking values in [0, 1].
    Depending on the choice of font, the text may or may not be anti-aliased.
    anti-aliased text can take any value between 0 and 1, while
    non-anti-aliased text produces a binary image.

    This method requires the ``pillow`` package.

    Parameters
    ----------
    space : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created.
        Must be two-dimensional.
    text : str
        The text that should be written onto the background.
    font : str, optional
        The font that should be used to write the text. Available options are
        platform dependent.
        Default: Platform dependent. 'arial' for windows,
        'LiberationSans-Regular' for linux and 'Helvetica' for OSX
    border : float, optional
        Padding added around the text. 0.0 indicates that the phantom should
        occupy all of the space along its largest dimension while 1.0 gives a
        maximally padded image (text not visible).
    inverted : bool, optional
        If the phantom should be given in inverted style, i.e. white on black.

    Returns
    -------
    phantom : ``space`` element
        The text phantom in ``space``.

    Notes
    -----
    The set of available fonts is highly platform dependent, and there is no
    obvious way (except from trial and error) to find what fonts are supported
    on an arbitrary platform.

    In general, the fonts ``'arial'``, ``'calibri'`` and ``'impact'`` tend to
    be available on windows.

    Platform dependent tricks:

    **Linux**::

        $ find /usr/share/fonts -name "*.[to]tf"
    """
    from PIL import Image, ImageDraw, ImageFont

    if space.ndim != 2:
        raise ValueError('`space` must be two-dimensional')

    if font is None:
        platform = sys.platform
        if platform == 'win32':
            # Windows
            font = 'arial'
        elif platform == 'darwin':
            # Mac OSX
            font = 'Helvetica'
        else:
            # Assume platform is linux
            font = 'LiberationSans-Regular'

    text = str(text)

    # Figure out what font size we should use by creating a very high
    # resolution font and calculating the size of the text in this font
    init_size = 1000
    init_pil_font = ImageFont.truetype(font + ".ttf", size=init_size,
                                       encoding="unic")
    init_text_width, init_text_height = init_pil_font.getsize(text)

    # True size is given by how much too large (or small) the example was
    scaled_init_size = (1.0 - border) * init_size
    size = scaled_init_size * min([space.shape[0] / init_text_width,
                                   space.shape[1] / init_text_height])
    size = int(size)

    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size,
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', space.shape, (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = ((space.shape[0] - text_width) // 2,
              (space.shape[1] - text_height) // 2)
    white = "#000000"
    draw.text(offset, text, font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    arr = np.asarray(canvas)
    arr = np.sum(arr, -1)
    arr = arr / np.max(arr)
    arr = np.rot90(arr, -1)

    if inverted:
        arr = 1 - arr

    return space.element(arr)

def disc_phantom(discr, smooth=True, taper=20.0):
    """Return a 'disc' phantom.

    This phantom is used in [Okt2015]_ for shape-based reconstruction.

    Parameters
    ----------
    discr : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created
    smooth : `bool`, optional
        If `True`, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : `float`, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : `DiscreteLpVector`
    """
    if discr.ndim == 2:
        if smooth:
            return _disc_phantom_2d_smooth(discr, taper)
        else:
            return _disc_phantom_2d_nonsmooth(discr)
    else:
        raise ValueError('Phantom only defined in 2 dimensions, got {}.'
                         ''.format(discr.dim))


def _disc_phantom_2d_smooth(discr, taper):
    """Return a 2d smooth 'disc' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_circle(x):
        """Blurred characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.2, 0.2)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    out = discr.element(blurred_circle)
    return out.ufuncs.minimum(1, out=out)


def _disc_phantom_2d_nonsmooth(discr):
    """Return a 2d nonsmooth 'disc' phantom."""

    def circle(x):
        """Characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.2, 0.2)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    out = discr.element(circle)
    return out.ufuncs.minimum(1, out=out)


def donut(discr, smooth=True, taper=20.0):
    """Return a 'donut' phantom.

    This phantom is used in [Okt2015]_ for shape-based reconstruction.

    Parameters
    ----------
    discr : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created
    smooth : `bool`, optional
        If `True`, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : `float`, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : `DiscreteLpElement`
    """
    if discr.ndim == 2:
        if smooth:
            return _donut_2d_smooth(discr, taper)
        else:
            return _donut_2d_nonsmooth(discr)
    else:
        raise ValueError('Phantom only defined in 2 dimensions, got {}.'
                         ''.format(discr.dim))


def _donut_2d_smooth(discr, taper):
    """Return a 2d smooth 'donut' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_circle_1(x):
        """Blurred characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.4, 0.4]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    def blurred_circle_2(x):
        """Blurred characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    out = discr.element(blurred_circle_1) - discr.element(blurred_circle_2)
    return out.ufuncs.minimum(1, out=out)


def _donut_2d_nonsmooth(discr):
    """Return a 2d nonsmooth 'donut' phantom."""

    def circle_1(x):
        """Characteristic function of an ellipse.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.4, 0.4]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    def circle_2(x):
        """Characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    out = discr.element(circle_1) - discr.element(circle_2)
    
    return out.ufuncs.minimum(1, out=out)


def sphere(discr, smooth=True, taper=20.0):
    """Return a 'sphere' phantom.

    Parameters
    ----------
    discr : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created
    smooth : `bool`, optional
        If `True`, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : `float`, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : `DiscreteLpElement`
    """
    if discr.ndim == 3:
        if smooth:
            return _sphere_3d_smooth(discr, taper)
        else:
            return _sphere_3d_nonsmooth(discr)
    else:
        raise ValueError('Phantom only defined in 3 dimensions, got {}.'
                         ''.format(discr.dim))


def _sphere_3d_smooth(discr, taper):
    """Return a 3d smooth 'sphere' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_sphere(x):
        """Blurred characteristic function of a sphere.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.1, 0.1, 0.1)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.05, 0.05, 0.05]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0, -0.15]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    out = discr.element(blurred_sphere)
    return out.ufuncs.minimum(1, out=out)


def _sphere_3d_nonsmooth(discr):
    """Return a 3d nonsmooth 'sphere' phantom."""

    def sphere(x):
        """Characteristic function of an ellipse.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.1, 0.1, 0.1)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.05, 0.05, 0.05]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0, -0.15]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    out = discr.element(sphere)
    return out.ufuncs.minimum(1, out=out)


def sphere2(discr, smooth=True, taper=20.0):
    """Return a 'sphere' phantom.

    Parameters
    ----------
    discr : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created
    smooth : `bool`, optional
        If `True`, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : `float`, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : `DiscreteLpElement`
    """
    if discr.ndim == 3:
        if smooth:
            return _sphere_3d_smooth2(discr, taper)
        else:
            return _sphere_3d_nonsmooth2(discr)
    else:
        raise ValueError('Phantom only defined in 3 dimensions, got {}.'
                         ''.format(discr.dim))


def _sphere_3d_smooth2(discr, taper):
    """Return a 3d smooth 'sphere' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_sphere(x):
        """Blurred characteristic function of a sphere.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.1, 0.1, 0.1)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    out = discr.element(blurred_sphere)
    return out.ufuncs.minimum(1, out=out)


def _sphere_3d_nonsmooth2(discr):
    """Return a 3d nonsmooth 'sphere' phantom."""

    def sphere(x):
        """Characteristic function of an ellipse.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.1, 0.1, 0.1)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    out = discr.element(sphere)
    return out.ufuncs.minimum(1, out=out)


def cube(space, smooth=True, taper=20.0):
    """Return a 3D 'cube' phantom.

    Parameters
    ----------
    space : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created.
    smooth : bool, optional
        If ``True``, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : float, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : ``space`` element
        The submarine phantom in ``space``.
    """
    if space.ndim == 3:
        if smooth:
            return _cube_3d_smooth(space, taper)
        else:
            return _cube_3d_nonsmooth(space)
    else:
        raise ValueError('phantom only defined in 3 dimensions, got {}'
                         ''.format(space.ndim))


def _cube_3d_smooth(space, taper):
    """Return a 2d smooth 'cube' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_cube(x):
        """Blurred characteristic function of a cube.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1] x [0, 1]``,
        the rect has lower left ``(0.35, 0.35, 0.35)`` and upper right
        ``(0.65, 0.65, 0.65)``. For other domains, the values are scaled
        accordingly.
        """
        xlower = np.array([0.35, 0.35, 0.35]) * space.domain.extent()
        xlower += space.domain.min()
        xupper = np.array([0.65, 0.65, 0.65]) * space.domain.extent()
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            length = upp - low
            out = out * (logistic((xi - low) / length, taper) *
                         logistic((upp - xi) / length, taper))
        return out

    out = space.element(blurred_cube)
    return out.ufuncs.minimum(1, out=out)


def _cube_3d_nonsmooth(space):
    """Return a 2d nonsmooth 'cube' phantom."""

    def cube(x):
        """Characteristic function of a rectangle.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1] x [0, 1]``,
        the rect has lower left ``(0.35, 0.35, 0.35)`` and upper right
        ``(0.65, 0.65, 0.65)``. For other domains, the values are scaled
        accordingly.
        """
        xlower = np.array([0.35, 0.35, 0.35]) * space.domain.extent()
        xlower += space.domain.min()
        xupper = np.array([0.65, 0.65, 0.65]) * space.domain.extent()
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            out = out * ((xi >= low) & (xi <= upp))
        return out

    out = space.element(cube)
    return out.ufuncs.minimum(1, out=out)


if __name__ == '__main__':
    # Show the phantoms
    import odl

    space = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    submarine(space, smooth=False).show('submarine smooth=False')
    submarine(space, smooth=True).show('submarine smooth=True')
    submarine(space, smooth=True, taper=50).show('submarine taper=50')


    text(space, text='phantom').show('phantom')
    disc_phantom(space, smooth=False).show('disc smooth=False')
    disc_phantom(space, smooth=True).show('disc smooth=True')
    disc_phantom(space, smooth=True, taper=50).show('disc taper=50')

    donut(space, smooth=False).show('donut smooth=False')
    donut(space, smooth=True).show('donut smooth=True')
    donut(space, smooth=True, taper=50).show('donut taper=50')
    
    space_3d = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [300, 300, 300])
    
    sphere(space_3d, smooth=False).show('sphere smooth=False',
          indices=np.s_[:, :, space_3d.shape[-1] // 2])
    sphere(space_3d, smooth=True).show('sphere smooth=True',
          indices=np.s_[space_3d.shape[-1] // 2, :, :])
    sphere(space_3d, smooth=True, taper=50).show('sphere taper=50',
          indices=np.s_[space_3d.shape[-1] // 2, :, :])
    
    sphere2(space_3d, smooth=False).show('sphere smooth=False',
          indices=np.s_[space_3d.shape[-1] // 2, :, :])
    sphere2(space_3d, smooth=True).show('sphere smooth=True',
          indices=np.s_[space_3d.shape[-1] // 2, :, :])
    sphere2(space_3d, smooth=True, taper=50).show('sphere taper=50',
          indices=np.s_[space_3d.shape[-1] // 2, :, :])
    
    cube(space_3d, smooth=False).show('cube smooth=False',
        indices=np.s_[space_3d.shape[-1] // 2, :, :])
    cube(space_3d, smooth=True).show('cube smooth=True',
        indices=np.s_[space_3d.shape[-1] // 2, :, :])
    cube(space_3d, smooth=True, taper=50).show('cube taper=50',
        indices=np.s_[space_3d.shape[-1] // 2, :, :])

    # Run also the doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
