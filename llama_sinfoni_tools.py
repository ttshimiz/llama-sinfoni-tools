# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:56:56 2016

@author: ttshimiz
"""

import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import astropy.modeling as apy_mod
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from spectral_cube import SpectralCube
import aplpy
import matplotlib.pyplot as plt
import lines

def read_data(fn):
    """

    Reads in SINFONI FITS cube, cleans up the header, and returns a
    spectral-cube object.

    Parameters
    ----------
    fn = string; FITS file names

    Returns
    -------
    cube = spectral_cube.SpectralCube object

    """

    data = fits.getdata(fn)*1e-17
    header = fits.getheader(fn)

    # Check the spectral axis units and values
    # Switch to meters if the unit is micron
    cunit3 = header['CUNIT3']
    crval3 = header['CRVAL3']
    cdelt3 = header['CDELT3']

    if cunit3 == 'MICRON':
        cunit3 = 'meter'
        crval3 = crval3*10**-6
        cdelt3 = cdelt3*10**-6

    header['CUNIT3'] = cunit3
    header['CRVAL3'] = crval3
    header['CDELT3'] = cdelt3

    wcs = WCS(header)

    # Check now the cdelt3 value in the WCS object
    if wcs.wcs.cd[2, 2] != cdelt3:
        wcs.wcs.cd[2, 2] = cdelt3

    cube = SpectralCube(data=data, wcs=wcs, read_beam=False, meta={'BUNIT':'W / (m2 micron)'})

    # Convert to microns
    cube = cube.with_spectral_unit(u.micron)

    return cube


def cont_fit_single(x, spectrum, degree=1, errors=None, exclude=None):
    """
    Function to fit the continuum of a single spectrum with a polynomial.
    """

    if errors is None:
        errors = np.ones(len(spectrum))

    if exclude is not None:
        x = x[~exclude]
        spectrum = spectrum[~exclude]
        errors = errors[~exclude]

    cont = apy_mod.models.Polynomial1D(degree=degree)

    # Use the endpoints of the spectrum to guess at zeroth and first order
    # parameters
    y1 = spectrum[0]
    y2 = spectrum[-1]
    x1 = x[0]
    x2 = x[-1]
    cont.c1 = (y2-y1)/(x2-x1)
    cont.c0 = y1 - cont.c1*x1

    fitter = apy_mod.fitting.LevMarLSQFitter()
    #cont_fit = fitter(cont, x, spectrum, weights=1./errors)
    cont_fit = fitter(cont, x, spectrum)

    return cont_fit


def remove_cont(cube, degree=1, exclude=None):
    """
    Function to loop through all of the spectra in a cube and subtract out the continuum
    """

    xsize = cube.shape[1]
    ysize = cube.shape[2]
    nparams = degree+1
    fit_params = np.zeros((xsize, ysize, nparams))
    spec_ax = cube.spectral_axis.value
    data_cont_remove = np.zeros(cube.shape)

    for i in range(xsize):
        for j in range(ysize):

            spec = cube[:, i, j].value/10**(-17)

            if np.any(~np.isnan(spec)):
                cont = cont_fit_single(spec_ax, spec, degree=degree, exclude=exclude)

                for n in range(nparams):
                    fit_params[i, j, n] = cont.parameters[n]

                data_cont_remove[:, i, j] = (spec - cont(spec_ax))*10**(-17)

            else:
                fit_params[i, j, :] = np.nan
                data_cont_remove[:, i, j] = np.nan

    cube_cont_remove = SpectralCube(data=data_cont_remove, wcs=cube.wcs,
                                    meta={'BUNIT':cube.unit.to_string()})
    cube_cont_remove = cube_cont_remove.with_spectral_unit(cube.spectral_axis.unit)

    return cube_cont_remove, fit_params


def calc_local_rms(cube, exclude=None):
    """
    Function to calculate the local rms of the spectrum around the line.
    Assumes the continuum has been subtracted already.
    Excludes the region around the line center +/- 'region'
    """

    xsize = cube.shape[1]
    ysize = cube.shape[2]
    flux_unit = cube.unit
    spec_ax = cube.spectral_axis
    #ind_use = ((spec_ax < (line_center+region)) & (spec_ax > (line_center-region)))
    local_rms = np.zeros((xsize, ysize))*flux_unit

    for i in range(xsize):
        for j in range(ysize):

            spec = cube[:, i, j].value
            if exclude is not None:
                local_rms[i, j] = np.std(spec[~exclude])*flux_unit
            else:
                local_rms[i, j] = np.std(spec)*flux_unit

    return local_rms


def calc_line_params(fit_params, line_centers, inst_broad=0):
    """
    Function to determine the integrated line flux, velocity, and linewidth
    Assumes the units on the amplitude are W/m^2/micron and the units on the
    mean and sigma are micron as well.
    Also determines the S/N of the line using the local rms of the spectrum.
    """

    line_params = {}

    for k in fit_params.keys():

        lc = line_centers[k]
        line_params[k] = {}
        amp = fit_params[k]['amplitude']
        line_mean = fit_params[k]['mean']
        line_sigma = fit_params[k]['sigma']

        if line_mean.unit != u.micron:
            print('Warning: Units on the line mean and sigma are not in microns.'
                  'Integrated line flux will not be correct.')

        # Integrated flux is just a Gaussian integral from -inf to inf
        int_flux = np.sqrt(2*np.pi)*amp*np.abs(line_sigma)

        # Convert the line mean and line sigma to km/s if not already
        if line_mean.unit.physical_type != 'speed':
            velocity = line_mean.to(u.km/u.s, equivalencies=u.doppler_optical(lc))
            veldisp = (line_mean+line_sigma).to(u.km/u.s, equivalencies=u.doppler_optical(line_mean))
        else:
            velocity = line_mean.to(u.km/u.s)
            veldisp = line_sigma.to(u.km/u.s)

        line_params[k]['int_flux'] = int_flux
        line_params[k]['velocity'] = velocity

        # Subtract off instrumental broadening
        phys_veldisp = np.sqrt(veldisp**2 - inst_broad**2)
        phys_veldisp[veldisp < inst_broad] = np.nan

        line_params[k]['veldisp'] = phys_veldisp

    return line_params


def plot_line_params(line_params, header, vel_min=-200., vel_max=200.,
                     vdisp_max=300.):
    """
    Function to plot the line intensity, velocity, and velocity dispersion in one figure
    """

    int_flux_hdu = fits.PrimaryHDU()
    velocity_hdu = fits.PrimaryHDU()
    veldisp_hdu = fits.PrimaryHDU()

    header['WCSAXES'] = 2
    header['NAXIS'] = 2
    header.remove('CDELT3')
    header.remove('CRVAL3')
    header.remove('CUNIT3')
    header.remove('CRPIX3')
    header.remove('CTYPE3')

    int_flux_hdu.header = header
    velocity_hdu.header = header
    veldisp_hdu.header = header

    int_flux_hdu.data = line_params['int_flux'].value
    velocity_hdu.data = line_params['velocity'].value
    veldisp_hdu.data = line_params['veldisp'].value

    fig = plt.figure(figsize=(18,6))

    ax_int = aplpy.FITSFigure(int_flux_hdu, figure=fig, subplot=(1,3,1))
    ax_vel = aplpy.FITSFigure(velocity_hdu, figure=fig, subplot=(1,3,2))
    ax_vdp = aplpy.FITSFigure(veldisp_hdu, figure=fig, subplot=(1,3,3))

    int_mn, int_med, int_sig = sigma_clipped_stats(line_params['int_flux'].value, iters=100)
    vel_mn, vel_med, vel_sig = sigma_clipped_stats(line_params['velocity'].value[np.abs(line_params['velocity'].value) < 1000.], iters=100)
    vdp_mn, vdp_med, vdp_sig = sigma_clipped_stats(line_params['veldisp'].value, iters=100)

    ax_int.show_colorscale(cmap='cubehelix', stretch='log', vmin=0, vmid=-np.nanmax(int_flux_hdu.data)/1000.)
    ax_vel.show_colorscale(cmap='RdBu_r', vmin=vel_min, vmax=vel_max)
    ax_vdp.show_colorscale(cmap='gist_heat', vmin=0, vmax=vdisp_max)

    ax_int.set_nan_color('k')
    ax_vel.set_nan_color('k')
    ax_vdp.set_nan_color('k')

    ax_int.show_colorbar()
    ax_vel.show_colorbar()
    ax_vdp.show_colorbar()

    ax_int.colorbar.set_axis_label_text(r'Flux [W m$^{-2}$]')
    ax_vel.colorbar.set_axis_label_text(r'Velocity [km s$^{-1}$]')
    ax_vdp.colorbar.set_axis_label_text(r'$\sigma_{\rm v}$ [km s$^{-1}$]')

    ax_int.set_axis_labels_ydisp(-30)
    ax_vel.hide_yaxis_label()
    ax_vel.hide_ytick_labels()
    ax_vdp.hide_yaxis_label()
    ax_vdp.hide_ytick_labels()

    fig.subplots_adjust(wspace=0.3)

    return fig, [ax_int, ax_vel, ax_vdp]


def create_line_ratio_map(line1, line2, header, cmap='cubehelix',
                          line1_name=None, line2_name=None):
    """
    Function to create a line ratio map. Map will be line1/line2.
    """

    lr_hdu = fits.PrimaryHDU()

    header['WCSAXES'] = 2
    header['NAXIS'] = 2
    header.remove('CDELT3')
    header.remove('CRVAL3')
    header.remove('CUNIT3')
    header.remove('CRPIX3')
    header.remove('CTYPE3')

    lr_hdu.header = header

    lr_hdu.data = line1/line2

    lr_fig = aplpy.FITSFigure(lr_hdu)
    lr_mn, lr_med, lr_sig = sigma_clipped_stats(line1/line2, iters=100)

    lr_fig.show_colorscale(cmap=cmap, vmin=0.0, vmax=lr_med+2*lr_sig)

    lr_fig.show_colorbar()

    if ((line1_name is not None) & (line2_name is not None)):

        lr_fig.colorbar.set_axis_label_text(line1_name+'/'+line2_name)

    lr_fig.set_axis_labels_ydisp(-30)

    return lr_fig


def create_model(line_centers, amp_guess=None,
                 center_guess=None, width_guess=None,
                 center_limits=None, width_limits=None,
                 center_fixed=None, width_fixed=None, line_names=None):
    """
    Function that allows for the creation of a generic model for a spectral region.
    Each line specified in 'line_names' must be included in the file 'lines.py'.
    Defaults for the amplitude guesses will be 1.0 for all lines.
    Defaults for the center guesses will be the observed wavelengths.
    Defaults for the line widths will be 100 km/s for narrow lines and 1000 km/s for the
    broad lines.
    All lines are considered narrow unless the name has 'broad' attached to the end of the name.
    """

    # Line_names can be a single string. If so convert it to a list
    if type(line_centers) == str:
        line_names = [line_centers]
    nlines = len(line_centers)

    # Create the default amplitude guesses for the lines if necessary
    if amp_guess is None:
        amp_guess = np.ones(nlines)

    # Create arrays to hold the default line center and width guesses
    if center_guess is None:
        center_guess = np.zeros(nlines)*u.km/u.s
    if width_guess is None:
        width_guess = np.ones(nlines)*100.*u.km/u.s

    # Create default line names
    if line_names is None:
        line_names = ['Line '+str(i+1) for i in range(nlines)]

    # Loop through each line and create a model
    mods = []
    for i,l in enumerate(line_names):

        # Equivalency to convert to/from wavelength from/to velocity
        opt_conv = u.doppler_optical(line_centers[i])

        # Convert the guesses for the line center and width to micron
        center_guess_i = center_guess[i].to(u.micron, equivalencies=opt_conv)
        if u.get_physical_type(width_guess.unit) == 'speed':
            width_guess_i = width_guess[i].to(u.micron, equivalencies=u.doppler_optical(center_guess_i)) - center_guess_i
        elif u.get_physical_type(width_guess.unit) == 'length':
            width_guess_i = width_guess[i].to(u.micron)
        center_guess_i = center_guess_i.value
        width_guess_i = width_guess_i.value

        # Create the single Gaussian line model for the emission line
        mod_single = apy_mod.models.Gaussian1D(mean=center_guess_i, amplitude=amp_guess[i],
                                               stddev=width_guess_i, name=l)

        # Set the constraints on the parameters if necessary
        mod_single.amplitude.min = 0      # always an emission line

        if center_limits is not None:
            if center_limits[i][0] is not None:
                mod_single.mean.min = center_limits[i][0].to(u.micron, equivalencies=opt_conv).value
            if center_limits[i][1] is not None:
                mod_single.mean.max = center_limits[i][1].to(u.micron, equivalencies=opt_conv).value

        if width_limits is not None:
            if width_limits[i][0] is not None:
                mod_single.stddev.min = width_limits[i][0].to(u.micron, equivalencies=opt_conv).value - line_center.value
            else:
                mod_single.stddev.min = 0         # can't have negative width
            if width_limits[i][1] is not None:
                mod_single.stddev.max = width_limits[i][1].to(u.micron, equivalencies=opt_conv).value - line_center.value

        # Set the fixed parameters
        if center_fixed is not None:
            mod_single.mean.fixed = center_fixed[i]
        if width_fixed is not None:
            mod_single.stddev.fixed = width_fixed[i]

        # Add to the model list
        mods.append(mod_single)

    # Create the combined model by adding all of the models together
    if nlines == 1:
        final_model = mods[0]
    else:
        final_model = mods[0]
        for m in mods[1:]:
            final_model += m

    return final_model

def cubefit(cube, model, skip=None, exclude=None):
    """
    Function to loop through all of the spectra in a cube and fit a model.
    """

    xsize = cube.shape[1]
    ysize = cube.shape[2]
    flux_unit = cube.unit
    spec_ax = cube.spectral_axis
    spec_ax_unit = cube.spectral_axis.unit

    fit_params = {}
    if hasattr(model, 'submodel_names'):

        for n in model.submodel_names:
            fit_params[n] = {'amplitude': np.zeros((xsize, ysize))*flux_unit*np.nan,
                             'mean': np.zeros((xsize, ysize))*spec_ax_unit*np.nan,
                             'sigma': np.zeros((xsize, ysize))*spec_ax_unit*np.nan}

    else:
        fit_params[model.name] = {'amplitude': np.zeros((xsize, ysize))*flux_unit*np.nan,
                                  'mean': np.zeros((xsize, ysize))*spec_ax_unit*np.nan,
                                  'sigma': np.zeros((xsize, ysize))*spec_ax_unit*np.nan}

    if skip is None:
        skip = np.zeros((xsize, ysize), dtype=np.bool)

    for i in range(xsize):
        for j in range(ysize):

            spec = cube[:, i, j].value/10**(-17)

            if (np.any(~np.isnan(spec)) & ~skip[i, j]):

                fit_result = specfit(spec_ax.to(u.micron).value, spec, model, exclude=exclude)

                if hasattr(model, 'submodel_names'):
                    for n in model.submodel_names:
                        fit_params[n]['amplitude'][i,j] = fit_result[n].amplitude.value*flux_unit*10**(-17)
                        fit_params[n]['mean'][i,j] = fit_result[n].mean.value*spec_ax_unit
                        fit_params[n]['sigma'][i,j] = fit_result[n].stddev.value*spec_ax_unit
                else:
                    fit_params[model.name]['amplitude'][i,j] = fit_result.amplitude.value*flux_unit*10**(-17)
                    fit_params[model.name]['mean'][i,j] = fit_result.mean.value*spec_ax_unit
                    fit_params[model.name]['sigma'][i,j] = fit_result.stddev.value*spec_ax_unit


    return fit_params


def specfit(x, fx, model, errors=None, exclude=None):
    """
    Function to fit a single spectrum with a model
    """

    if errors is None:
        errors = np.ones(len(fx))
    if exclude is not None:
        x = x[~exclude]
        fx = fx[~exclude]
        errors = errors[~exclude]

    fitter = apy_mod.fitting.LevMarLSQFitter()
    bestfit = fitter(model, x, fx, weights=1./errors)

    return bestfit


def skip_pixels(cube, rms, sn_thresh=3.0):
    """
    Function to determine which pixels to skip based on a user defined S/N threshold.
    Returns an NxM boolean array where True indicates a pixel to skip.
    The signal used is the maximum value in the spectrum.
    If the maximum value is a NaN then that pixel is also skipped.
    """

    spec_max = cube.max(axis=0)
    sig_to_noise = spec_max/rms
    skip = (sig_to_noise.value < sn_thresh) | (np.isnan(sig_to_noise.value))

    return skip

def prepare_cube(cube, slice_center, velrange=[-4000., 4000.]*u.km/u.s):
    """
    Function to slice the cube and extract a specific spectral region
    based on a user-defined central wavelength and velocity range.
    """

    # Slice first based on velocity
    slice = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical',
                                    rest_value=slice_center).spectral_slab(velrange[0], velrange[1])

    # Convert back to wavelength
    slice = slice.with_spectral_unit(unit=u.micron, velocity_convention='optical',
                                     rest_value=slice_center)

    return slice


def runfit(cube, model, sn_thresh=3.0, cont_exclude=None, fit_exclude=None):

    # Subtract out the continuum
    cube_cont_remove, cont_params = remove_cont(cube, exclude=cont_exclude)

    # Determine the RMS around the line
    local_rms = calc_local_rms(cube_cont_remove, exclude=cont_exclude)

    # Create a mask of pixels to skip in the fitting
    skippix = skip_pixels(cube_cont_remove, local_rms, sn_thresh=sn_thresh)

    fit_params = cubefit(cube_cont_remove, fitmod, skip=skippix, exclude=fit_exclude)

    results = {'continuum_sub': cube_cont_remove,
               'cont_params': cont_params,
               'fit_params': fit_params,
               'fit_pixels': skippix}

    return results
