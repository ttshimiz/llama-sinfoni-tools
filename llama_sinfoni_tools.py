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


def gauss_fit_single(x, spectrum, rms_est=None, sn_thresh=None,
                     guess=None, errors=None, exclude=None):
    """
    Function to fit a single spectrum with a Gaussian
    """

    if errors is None:
         errors = np.ones(len(spectrum))

    model = apy_mod.models.Gaussian1D()
    model.amplitude.min = 0
    model.stddev.min = 0

    if guess is None:
        model.amplitude = np.max(spectrum)
        model.mean = x[np.argmax(spectrum)]
        model.stddev = 3*(x[1] - x[0])
    else:
        model.amplitude = guess[0]
        model.mean = guess[1]
        model.stddev = guess[2]

    if ((rms_est is not None) & (sn_thresh is not None)):
        sn_est = model.amplitude/rms_est
        if sn_est > sn_thresh:
            fitter = apy_mod.fitting.LevMarLSQFitter()
            gauss_fit = fitter(model, x, spectrum, weights=1./errors)
        else:
            gauss_fit = None
    else:
        fitter = apy_mod.fitting.LevMarLSQFitter()
        gauss_fit = fitter(model, x, spectrum, weights=1./errors)

    return gauss_fit


def gauss_fit_narrow_broad(x, spectrum, rms_est=None, sn_thresh=None,
                           guess=None, errors=None, exclude=None):
    """
    Function to fit a single spectrum with a Gaussian
    """

    if errors is None:
         errors = np.ones(len(spectrum))

    if exclude is not None:
        x = x[~exclude]
        spectrum = spectrum[~exclude]
        errors = errors[~exclude]

    narrowModel = apy_mod.models.Gaussian1D().rename('Narrow')
    narrowModel.amplitude.min = 0
    narrowModel.stddev.min = 0

    broadModel = apy_mod.models.Gaussian1D().rename('Broad')
    broadModel.amplitude.min = 0
    broadModel.stddev.min = 0

    if guess is None:
        narrowModel.amplitude = np.max(spectrum)
        narrowModel.mean = x[np.argmax(spectrum)]
        narrowModel.stddev = 3*(x[1] - x[0])

        broadModel.amplitude = narrowModel.amplitude/2.
        broadModel.mean = narrowModel.mean
        broadModel.stddev = narrowModel.stddev*10.

    else:
        narrowmodel.amplitude = guess[0]
        narrowmodel.mean = guess[1]
        narrowmodel.stddev = guess[2]

        broadModel.amplitude = guess[3]
        broadModel.mean = guess[4]
        broadModel.stddev = guess[5]

    model = narrowModel + broadModel

    if ((rms_est is not None) & (sn_thresh is not None)):
        sn_est = model.amplitude_0/rms_est
        if sn_est > sn_thresh:
            fitter = apy_mod.fitting.LevMarLSQFitter()
            gauss_fit = fitter(model, x, spectrum, weights=1./errors)
        else:
            gauss_fit = None
    else:
        fitter = apy_mod.fitting.LevMarLSQFitter()
        gauss_fit = fitter(model, x, spectrum, weights=1./errors)

    return gauss_fit


def cubefit_gauss(cube, local_rms=None, sn_thresh=None, guess=None, exclude=None):
    """
    Function to loop through all of the spectra in a cube and fit a gaussian.
    If an estimate of the local rms and a signal-to-noise threshold is given
    then only fit a line above the signal-to-noise threshold determined by
    the ratio of the local rms and the peak value in the spectrum.
    """

    xsize = cube.shape[1]
    ysize = cube.shape[2]
    flux_unit = cube.unit
    spec_ax = cube.spectral_axis.value
    spec_ax_unit = cube.spectral_axis.unit

    fit_params = {'amplitude': np.zeros((xsize, ysize))*flux_unit,
                  'mean': np.zeros((xsize, ysize))*spec_ax_unit,
                  'sigma': np.zeros((xsize, ysize))*spec_ax_unit}

    for i in range(xsize):
        for j in range(ysize):

            spec = cube[:, i, j].value/10**(-17)
            if local_rms is not None:
                rms_est = local_rms[i, j].value/10**(-17)
            else:
                rms_est = None

            if np.any(~np.isnan(spec)):
                gauss_mod = gauss_fit_single(spec_ax, spec, rms_est=rms_est,
                                             sn_thresh=sn_thresh,guess=guess,
                                             exclude=exclude)
                if gauss_mod is not None:
                    fit_params['amplitude'][i, j] = gauss_mod.amplitude.value*flux_unit*10**(-17)
                    fit_params['mean'][i, j] = gauss_mod.mean.value*spec_ax_unit
                    fit_params['sigma'][i, j] = gauss_mod.stddev.value*spec_ax_unit
                else:
                    fit_params['amplitude'][i, j] = np.nan
                    fit_params['mean'][i, j] = np.nan
                    fit_params['sigma'][i, j] = np.nan
            else:
                fit_params['amplitude'][i, j] = np.nan
                fit_params['mean'][i, j] = np.nan
                fit_params['sigma'][i, j] = np.nan

    return fit_params


def cubefit_broad(cube, local_rms=None, sn_thresh=None, guess=None, exclude=None):
    """
    Function to loop through all of the spectra in a cube and fit a combined narrow
    and broad component model.
    If an estimate of the local rms and a signal-to-noise threshold is given
    then only fit a line above the signal-to-noise threshold determined by
    the ratio of the local rms and the peak value in the spectrum.
    """

    xsize = cube.shape[1]
    ysize = cube.shape[2]
    flux_unit = cube.unit
    spec_ax = cube.spectral_axis.value
    spec_ax_unit = cube.spectral_axis.unit

    narrow_fit_params = {'amplitude': np.zeros((xsize, ysize))*flux_unit,
                         'mean': np.zeros((xsize, ysize))*spec_ax_unit,
                         'sigma': np.zeros((xsize, ysize))*spec_ax_unit}
    broad_fit_params = {'amplitude': np.zeros((xsize, ysize))*flux_unit,
                        'mean': np.zeros((xsize, ysize))*spec_ax_unit,
                        'sigma': np.zeros((xsize, ysize))*spec_ax_unit}

    for i in range(xsize):
        for j in range(ysize):

            spec = cube[:, i, j].value/10**(-17)
            if local_rms is not None:
                rms_est = local_rms[i, j].value/10**(-17)
            else:
                rms_est = None

            if np.any(~np.isnan(spec)):
                gauss_mod = gauss_fit_narrow_broad(spec_ax, spec, rms_est=rms_est,
                                                   sn_thresh=sn_thresh,guess=guess,
                                                   exclude=exclude)
                if gauss_mod is not None:
                    narrow_fit_params['amplitude'][i, j] = gauss_mod['Narrow'].amplitude.value*flux_unit*10**(-17)
                    narrow_fit_params['mean'][i, j] = gauss_mod['Narrow'].mean.value*spec_ax_unit
                    narrow_fit_params['sigma'][i, j] = gauss_mod['Narrow'].stddev.value*spec_ax_unit

                    broad_fit_params['amplitude'][i, j] = gauss_mod['Broad'].amplitude.value*flux_unit*10**(-17)
                    broad_fit_params['mean'][i, j] = gauss_mod['Broad'].mean.value*spec_ax_unit
                    broad_fit_params['sigma'][i, j] = gauss_mod['Broad'].stddev.value*spec_ax_unit

                else:
                    narrow_fit_params['amplitude'][i, j] = np.nan
                    narrow_fit_params['mean'][i, j] = np.nan
                    narrow_fit_params['sigma'][i, j] = np.nan

                    broad_fit_params['amplitude'][i, j] = np.nan
                    broad_fit_params['mean'][i, j] = np.nan
                    broad_fit_params['sigma'][i, j] = np.nan
            else:
                narrow_fit_params['amplitude'][i, j] = np.nan
                narrow_fit_params['mean'][i, j] = np.nan
                narrow_fit_params['sigma'][i, j] = np.nan

                broad_fit_params['amplitude'][i, j] = np.nan
                broad_fit_params['mean'][i, j] = np.nan
                broad_fit_params['sigma'][i, j] = np.nan


    return narrow_fit_params, broad_fit_params


def calc_local_rms(cube, line_center, exclude=None):
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


def calc_line_params(fit_params, line_center, local_rms=None, inst_broad=0):
    """
    Function to determine the integrated line flux, velocity, and linewidth
    Assumes the units on the amplitude are W/m^2/micron and the units on the
    mean and sigma are micron as well.
    Also determines the S/N of the line using the local rms of the spectrum.
    """

    amp = fit_params['amplitude']
    line_mean = fit_params['mean']
    line_sigma = fit_params['sigma']
    line_params = {}

    if line_mean.unit != u.micron:
        print('Warning: Units on the line mean and sigma are not in microns.'
              'Integrated line flux will not be correct.')

    # Integrated flux is just a Gaussian integral from -inf to inf
    int_flux = np.sqrt(2*np.pi)*amp*np.abs(line_sigma)

    # Convert the line mean and line sigma to km/s if not already
    if line_mean.unit.physical_type != 'speed':
        velocity = line_mean.to(u.km/u.s, equivalencies=u.doppler_optical(line_center))
        veldisp = (line_mean+line_sigma).to(u.km/u.s, equivalencies=u.doppler_optical(line_mean))
    else:
        velocity = line_mean.to(u.km/u.s)
        veldisp = line_sigma.to(u.km/u.s)

    line_params['int_flux'] = int_flux
    line_params['velocity'] = velocity

    # Subtract off instrumental broadening
    phys_veldisp = np.sqrt(veldisp**2 - inst_broad**2)
    phys_veldisp[veldisp < inst_broad] = np.nan

    line_params['veldisp'] = phys_veldisp

    return line_params


def plot_line_params(line_params, header):
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
    ax_vel.show_colorscale(cmap='RdBu_r', vmin=vel_med-2*vel_sig, vmax=vel_med+2*vel_sig)
    ax_vdp.show_colorscale(cmap='gist_heat', vmin=0, vmax=vdp_med+2*vdp_sig)

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


def run_line(cube, line_name, velrange =[-4000, 4000],
             zz=0, inst_broad=0., plot_results=True, sn_thresh=3.0):

    # Get the rest wavelength
    line_center = lines.EMISSION_LINES[line_name]*(1+zz)

    # Slice the cube
    slice = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical',
                                    rest_value=line_center).spectral_slab(velrange[0]*u.km/u.s, velrange[1]*u.km/u.s)
    slice = slice.with_spectral_unit(unit=u.micron, velocity_convention='optical',
                                     rest_value=line_center)

    # Subtract out the continuum
    cube_cont_remove, cont_params = remove_cont(slice)

    # Determine the RMS around the line
    local_rms = calc_local_rms(cube_cont_remove, line_center)

    # Fit a Gaussian to the line
    gaussfit_params = cubefit_gauss(cube_cont_remove, local_rms=local_rms, sn_thresh=sn_thresh)

    # Calculate the line parameters
    line_params = calc_line_params(gaussfit_params, line_center, local_rms, inst_broad=inst_broad)

    results = {'line_params': line_params,
               'continuum_sub': cube_cont_remove,
               'gauss_params': gaussfit_params,
               'data': slice}

    if plot_results:
        fig, axes = plot_line_params(line_params, slice.header)
        results['results_fig'] = fig
        results['results_axes'] = axes

    return results
