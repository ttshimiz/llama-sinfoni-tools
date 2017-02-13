# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:56:56 2016

@author: ttshimiz
"""

import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import astropy.modeling as apy_mod
import astropy.convolution as apy_conv
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.wcs import WCS
from spectral_cube import SpectralCube
import aplpy
import matplotlib.pyplot as plt
import lines
import multiprocessing
import time
import peakutils

def read_data(fn, scale=True):
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

    if scale:
        data = fits.getdata(fn)*1e-17
    else:
        data = fits.getdata(fn)

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
    try:
        if wcs.wcs.cd[2, 2] != cdelt3:
            wcs.wcs.cd[2, 2] = cdelt3
    except AttributeError:
        pass

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

    # Initialize the main fitter and the fitter that implements outlier removal using
    # sigma clipping. Default is to do 5 iterations removing all 3-sigma outliers
    fitter = apy_mod.fitting.LevMarLSQFitter()
    or_fitter = apy_mod.fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=5, sigma=3.0)
    filtered_data, cont_fit = or_fitter(cont, x, spectrum)

    return cont_fit


def remove_cont(cube, degree=1, exclude=None):
    """
    Function to loop through all of the spectra in a cube and subtract out the continuum
    """

    xsize = cube.shape[1]
    ysize = cube.shape[2]
    nparams = degree+1
    fit_params = np.zeros((nparams, xsize, ysize))
    spec_ax = cube.spectral_axis.value
    data_cont_remove = np.zeros(cube.shape)

    for i in range(xsize):
        for j in range(ysize):

            spec = cube[:, i, j].value/10**(-17)

            if np.any(~np.isnan(spec)):
                cont = cont_fit_single(spec_ax, spec, degree=degree, exclude=exclude)

                for n in range(nparams):
                    fit_params[n, i, j] = cont.parameters[n]

                data_cont_remove[:, i, j] = (spec - cont(spec_ax))*10**(-17)

            else:
                fit_params[:,i, j] = np.nan
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


def calc_line_params(fit_params, line_centers, fit_params_mc=None, inst_broad=0):
    """
    Function to determine the integrated line flux, velocity, and linewidth
    Assumes the units on the amplitude are W/m^2/micron and the units on the
    mean and sigma are micron as well.
    If there are parameters from a Monte Carlo session, use these to determine errors
    on the flux, velocity, and velocity dispersion.
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
        if fit_params_mc is not None:
            amp_mc = fit_params_mc[k]['amplitude']
            mean_mc = fit_params_mc[k]['mean']
            sigma_mc = fit_params_mc[k]['sigma']
            int_flux_mc = np.sqrt(2*np.pi)*amp_mc*np.abs(sigma_mc)

        # Convert the line mean and line sigma to km/s if not already
        if line_mean.unit.physical_type != 'speed':
            velocity = line_mean.to(u.km/u.s, equivalencies=u.doppler_optical(lc))
            veldisp = (line_mean+line_sigma).to(u.km/u.s, equivalencies=u.doppler_optical(line_mean))

            if fit_params_mc is not None:
                velocity_mc = mean_mc.to(u.km/u.s, equivalencies=u.doppler_optical(lc))
                veldisp_mc = (mean_mc+sigma_mc).to(u.km/u.s, equivalencies=u.doppler_optical(mean_mc))
        else:
            velocity = line_mean.to(u.km/u.s)
            veldisp = line_sigma.to(u.km/u.s)

            if fit_params_mc is not None:
                velocity_mc = mean_mc.to(u.km/u.s)
                veldisp_mc = sigma_mc.to(u.km/u.s)

        line_params[k]['int_flux'] = int_flux
        line_params[k]['velocity'] = velocity

        # Subtract off instrumental broadening
        phys_veldisp = np.sqrt(veldisp**2 - inst_broad**2)
        phys_veldisp[veldisp < inst_broad] = 0.*u.km/u.s

        line_params[k]['veldisp'] = phys_veldisp

        if fit_params_mc is not None:
            int_flux_err = np.zeros(int_flux.shape)*int_flux.unit
            vel_err = np.zeros(velocity.shape)*velocity.unit
            veldisp_err = np.zeros(veldisp.shape)*veldisp.unit

            for i in range(int_flux.shape[0]):
                for j in range(int_flux.shape[1]):
                    int_flux_err[i, j] = np.nanstd(int_flux_mc[:, i, j].value)*int_flux.unit
                    vel_err[i, j] = np.nanstd(velocity_mc[:, i, j].value)*velocity.unit
                    veldisp_err[i, j] = np.nanstd(veldisp_mc[:, i, j].value)*veldisp.unit

            line_params[k]['int_flux_err'] = int_flux_err
            line_params[k]['velocity_err'] = vel_err
            phys_veldisp_err = veldisp*veldisp_err/phys_veldisp
            line_params[k]['veldisp_err'] = phys_veldisp_err

    return line_params


def plot_line_params(line_params, header=None, vel_min=-200., vel_max=200.,
                     vdisp_max=300., mask=None, flux_scale='arcsinh'):
    """
    Function to plot the line intensity, velocity, and velocity dispersion in one figure
    """

    int_flux_hdu = fits.PrimaryHDU()
    velocity_hdu = fits.PrimaryHDU()
    veldisp_hdu = fits.PrimaryHDU()

    if (header is not None):
        header['WCSAXES'] = 2
        header['NAXIS'] = 2
        try:
            header.remove('CDELT3')
        except KeyError:
            pass
        try:
            header.remove('CRVAL3')
        except KeyError:
            pass
        try:
            header.remove('CUNIT3')
        except KeyError:
            pass
        try:
            header.remove('CRPIX3')
        except KeyError:
            pass
        try:
            header.remove('CTYPE3')
        except KeyError:
            pass

        int_flux_hdu.header = header
        velocity_hdu.header = header
        veldisp_hdu.header = header

    int_flux_hdu.data = line_params['int_flux'].value.copy()
    velocity_hdu.data = line_params['velocity'].value.copy()
    veldisp_hdu.data = line_params['veldisp'].value.copy()

    if mask is not None:
        int_flux_hdu.data[mask] = np.nan
        velocity_hdu.data[mask] = np.nan
        veldisp_hdu.data[mask] = np.nan

    fig = plt.figure(figsize=(18,6))

    ax_int = aplpy.FITSFigure(int_flux_hdu, figure=fig, subplot=(1,3,1))
    ax_vel = aplpy.FITSFigure(velocity_hdu, figure=fig, subplot=(1,3,2))
    ax_vdp = aplpy.FITSFigure(veldisp_hdu, figure=fig, subplot=(1,3,3))

    #int_mn, int_med, int_sig = sigma_clipped_stats(line_params['int_flux'].value, iters=100)
    #vel_mn, vel_med, vel_sig = sigma_clipped_stats(line_params['velocity'].value[np.abs(line_params['velocity'].value) < 1000.], iters=100)
    #vdp_mn, vdp_med, vdp_sig = sigma_clipped_stats(line_params['veldisp'].value, iters=100)

    ax_int.show_colorscale(cmap='cubehelix', stretch=flux_scale, vmin=0, vmid=-np.nanmax(int_flux_hdu.data)/1000.)
    ax_vel.show_colorscale(cmap='RdBu_r', vmin=vel_min, vmax=vel_max)
    ax_vdp.show_colorscale(cmap='inferno', vmin=0, vmax=vdisp_max)

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
                 center_fixed=None, width_fixed=None):
    """
    Function that allows for the creation of a generic model for a spectral region.
    Each line specified in 'line_names' must be included in the file 'lines.py'.
    Defaults for the amplitude guesses will be 1.0 for all lines.
    Defaults for the center guesses will be the observed wavelengths.
    Defaults for the line widths will be 100 km/s for narrow lines and 1000 km/s for the
    broad lines.
    All lines are considered narrow unless the name has 'broad' attached to the end of the name.
    """


    nlines = len(line_centers.keys())
    line_names = line_centers.keys()

    # Create the default amplitude guesses for the lines if necessary
    if amp_guess is None:
        amp_guess = {l:1.0 for l in line_names}

    # Create arrays to hold the default line center and width guesses
    if center_guess is None:
        center_guess = {l:0*u.km/u.s for l in line_names}
    if width_guess is None:
        width_guess = {l:100.*u.km/u.s for l in line_names}

    # Loop through each line and create a model
    mods = []
    for i,l in enumerate(line_names):

        # Equivalency to convert to/from wavelength from/to velocity
        opt_conv = u.doppler_optical(line_centers[l])

        # Convert the guesses for the line center and width to micron
        center_guess_i = center_guess[l].to(u.micron, equivalencies=opt_conv)
        if u.get_physical_type(width_guess[l].unit) == 'speed':
            width_guess_i = width_guess[l].to(u.micron, equivalencies=u.doppler_optical(center_guess_i)) - center_guess_i
        elif u.get_physical_type(width_guess[l].unit) == 'length':
            width_guess_i = width_guess[i].to(u.micron)
        center_guess_i = center_guess_i.value
        width_guess_i = width_guess_i.value

        # Create the single Gaussian line model for the emission line
        mod_single = apy_mod.models.Gaussian1D(mean=center_guess_i, amplitude=amp_guess[l],
                                               stddev=width_guess_i, name=l)

        # Set the constraints on the parameters if necessary
        mod_single.amplitude.min = 0      # always an emission line

        if center_limits is not None:
            if center_limits[l][0] is not None:
                mod_single.mean.min = center_limits[l][0].to(u.micron, equivalencies=opt_conv).value
            if center_limits[l][1] is not None:
                mod_single.mean.max = center_limits[l][1].to(u.micron, equivalencies=opt_conv).value

        if width_limits is not None:
            if width_limits[l][0] is not None:
                mod_single.stddev.min = width_limits[l][0].to(u.micron, equivalencies=opt_conv).value - line_centers[l].value
            else:
                mod_single.stddev.min = 0         # can't have negative width
            if width_limits[l][1] is not None:
                mod_single.stddev.max = width_limits[l][1].to(u.micron, equivalencies=opt_conv).value - line_centers[l].value
        else:
            mod_single.stddev.min = 0

        # Set the fixed parameters
        if center_fixed is not None:
            mod_single.mean.fixed = center_fixed[l]
        if width_fixed is not None:
            mod_single.stddev.fixed = width_fixed[l]

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

def cubefit(cube, model, skip=None, exclude=None, line_centers=None,
            auto_guess=False, guess_type=None, guess_region=None,
            calc_uncert=False, nmc=100., rms=None, parallel=False, cores=None):
    """
    Function to loop through all of the spectra in a cube and fit a model.
    """

    xsize = cube.shape[1]
    ysize = cube.shape[2]
    flux_unit = cube.unit
    spec_ax = cube.spectral_axis
    lam = spec_ax.to(u.micron).value
    spec_ax_unit = cube.spectral_axis.unit
    residuals = np.zeros(cube.shape)

    fit_params = {}

    if skip is None:
        skip = np.zeros((xsize, ysize), dtype=np.bool)

    print "Starting 'cubefit' with {0} spectral points and {1}x{2} image shape" .format(cube.shape[0], xsize, ysize)
    print "Total number of spaxels to fit: {0}/{1}".format(np.int(np.sum(~skip)), xsize*ysize)

    if calc_uncert:
        fit_params_mc = {}

    if hasattr(model, 'submodel_names'):

        for n in model.submodel_names:
            fit_params[n] = {'amplitude': np.zeros((xsize, ysize))*flux_unit*np.nan,
                             'mean': np.zeros((xsize, ysize))*spec_ax_unit*np.nan,
                             'sigma': np.zeros((xsize, ysize))*spec_ax_unit*np.nan}
            if calc_uncert:
               fit_params_mc[n] = {'amplitude': np.zeros((nmc, xsize, ysize))*flux_unit*np.nan,
                             'mean': np.zeros((nmc, xsize, ysize))*spec_ax_unit*np.nan,
                             'sigma': np.zeros((nmc, xsize, ysize))*spec_ax_unit*np.nan}

    else:
        fit_params[model.name] = {'amplitude': np.zeros((xsize, ysize))*flux_unit*np.nan,
                                  'mean': np.zeros((xsize, ysize))*spec_ax_unit*np.nan,
                                  'sigma': np.zeros((xsize, ysize))*spec_ax_unit*np.nan}
        if calc_uncert:
           fit_params_mc[model.name] = {'amplitude': np.zeros((nmc, xsize, ysize))*flux_unit*np.nan,
                                         'mean': np.zeros((nmc, xsize, ysize))*spec_ax_unit*np.nan,
                                         'sigma': np.zeros((nmc, xsize, ysize))*spec_ax_unit*np.nan}

    print "Lines being fit: {0}".format(fit_params.keys())

    if calc_uncert:
        print "Calculating uncertainties using MC simulation with {0} iterations.".format(nmc)
    else:
        print "No calculation of uncertainties."
    print "Starting fitting..."
    for i in range(xsize):
        for j in range(ysize):

            spec = cube[:, i, j].value/10**(-17)
            if calc_uncert:
                rms_i = rms[i, j].value/10**(-17)
            else:
                rms_i = None

            if (np.any(~np.isnan(spec)) & ~skip[i, j]):

                if auto_guess:

                    # Use the bounds on the line center as the guess region for each line
                    if guess_type == 'limits':
                        if hasattr(model, 'submodel_names'):
                            for k in fit_params.keys():
                                min_lam = model[k].mean.min
                                max_lam = model[k].mean.max
                                guess_region_line = (lam >= min_lam) & (lam <= max_lam)

                                ind_max = np.argmax(spec[guess_region_line])
                                wave_max = lam[guess_region_line][ind_max]
                                flux_max = spec[guess_region_line][ind_max]

                                model[k].mean = wave_max
                                model[k].amplitude = flux_max
                        else:
                            min_lam = model.mean.min
                            max_lam = model.mean.max
                            guess_region_line = (lam >= min_lam) & (lam <= max_lam)

                            ind_max = np.argmax(spec[guess_region_line])
                            wave_max = lam[guess_region_line][ind_max]
                            flux_max = spec[guess_region_line][ind_max]

                            model.mean = wave_max
                            model.amplitude = flux_max

                    elif guess_type == 'peak-find':

                        if guess_region is None:
                            guess_region = np.ones(len(spec), dtype=np.bool)

                        model = findpeaks(spec, lam, model, guess_region, line_centers)

                fit_results = specfit(lam, spec, model, exclude=exclude,
                                      calc_uncert=calc_uncert, nmc=nmc, rms=rms_i,
                                      parallel=parallel)
                print "Pixel {0},{1} fit successfully.".format(i,j)
                if calc_uncert:
                    best_fit = fit_results[0]
                    err_fits = fit_results[1]
                else:
                    best_fit = fit_results

                if hasattr(model, 'submodel_names'):
                    for n in model.submodel_names:
                        fit_params[n]['amplitude'][i,j] = best_fit[n].amplitude.value*flux_unit*10**(-17)
                        fit_params[n]['mean'][i,j] = best_fit[n].mean.value*spec_ax_unit
                        fit_params[n]['sigma'][i,j] = best_fit[n].stddev.value*spec_ax_unit

                        if calc_uncert:
                            mc_amps = np.array([err_fits[k][n].amplitude.value for k in range(nmc)])
                            mc_mean = np.array([err_fits[k][n].mean.value for k in range(nmc)])
                            mc_sig = np.array([err_fits[k][n].stddev.value for k in range(nmc)])

                            fit_params_mc[n]['amplitude'][:,i,j] = mc_amps*flux_unit*10**(-17)
                            fit_params_mc[n]['mean'][:,i,j] = mc_mean*spec_ax_unit
                            fit_params_mc[n]['sigma'][:,i,j] = mc_sig*spec_ax_unit
                else:
                    fit_params[model.name]['amplitude'][i,j] = best_fit.amplitude.value*flux_unit*10**(-17)
                    fit_params[model.name]['mean'][i,j] = best_fit.mean.value*spec_ax_unit
                    fit_params[model.name]['sigma'][i,j] = best_fit.stddev.value*spec_ax_unit

                    if calc_uncert:
                        mc_amps = np.array([err_fits[k].amplitude.value for k in range(nmc)])
                        mc_mean = np.array([err_fits[k].mean.value for k in range(nmc)])
                        mc_sig = np.array([err_fits[k].stddev.value for k in range(nmc)])

                        fit_params_mc[model.name]['amplitude'][:,i,j] = mc_amps*flux_unit*10**(-17)
                        fit_params_mc[model.name]['mean'][:,i,j] = mc_mean*spec_ax_unit
                        fit_params_mc[model.name]['sigma'][:,i,j] = mc_sig*spec_ax_unit

                residuals[:,i,j] = (spec - best_fit(spec_ax.to(u.micron).value))*10**(-17)
            else:
                print "Pixel {0},{1} skipped.".format(i,j)
                residuals[:,i,j] = spec*10**(-17)

    resid_cube = SpectralCube(data=residuals, wcs=cube.wcs,
                              meta={'BUNIT':cube.unit.to_string()})
    resid_cube = resid_cube.with_spectral_unit(cube.spectral_axis.unit)

    if calc_uncert:
        return [fit_params, resid_cube, fit_params_mc]
    else:
        return [fit_params, resid_cube]


def specfit(x, fx, model, exclude=None, calc_uncert=False, rms=None, nmc=100,
            parallel=False, cores=None):
    """
    Function to fit a single spectrum with a model.
    Option to run a Monte Carlo simulation to determine the uncertainties using
    either a simple for-loop or parallel.
    """

    if exclude is not None:
        x = x[~exclude]
        fx = fx[~exclude]

    bestfit = specfit_single(x, fx, model)

    if (calc_uncert) & (~parallel):

        rand_fits = []

        for i in range(nmc):
            rand_spec = np.random.randn(len(fx))*rms + fx
            rand_fit_i = specfit_single(x, rand_spec, model)
            rand_fits.append(rand_fit_i)

    elif (calc_uncert) & (parallel):

        rand_fits = specfit_parallel(x, fx, model, rms, nmc=nmc, cores=cores)

    if calc_uncert:
        return [bestfit, rand_fits]
    else:
        return bestfit


def specfit_parallel(x, fx, model, rms, nmc=100, cores=None):

    pool = multiprocessing.Pool(cores)

    mc_spec = np.zeros((nmc, len(fx)))
    for i in range(nmc):
        mc_spec[i, :] = np.random.randn(len(fx))*rms + fx
    result = [pool.apply_async(specfit_single, args=(x, mc_spec[i, :], model)) for i in range(nmc)]
    result = [p.get() for p in result]

    pool.close()

    return result


def specfit_single(x, fx, model):

    fitter = apy_mod.fitting.LevMarLSQFitter()
    modfit = fitter(model, x, fx)

    return modfit

def skip_pixels(cube, rms, sn_thresh=3.0, spec_use=None):
    """
    Function to determine which pixels to skip based on a user defined S/N threshold.
    Returns an NxM boolean array where True indicates a pixel to skip.
    The signal used is the maximum value in the spectrum.
    If the maximum value is a NaN then that pixel is also skipped.
    """

    if spec_use is None:
        spec_max = cube.max(axis=0)
        sig_to_noise = spec_max/rms
        skip = (sig_to_noise.value < sn_thresh) | (np.isnan(sig_to_noise.value))
    else:
        xsize = cube.shape[1]
        ysize = cube.shape[2]
        skip = np.zeros((xsize, ysize), dtype=np.bool)

        for x in range(xsize):
            for y in range(ysize):
                s = cube[:,x,y]
                s_n = np.max(s[spec_use])/rms[x,y]
                skip[x,y] = (s_n.value < sn_thresh) | (np.isnan(s_n.value))


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


def runfit(cube, model, sn_thresh=3.0, line_centers=None, cont_exclude=None, fit_exclude=None,
           max_guess=False, auto_guess=False, guess_type=None, guess_region=None, calc_uncert=False,
           nmc=100, cores=None, parallel=False):

    # Subtract out the continuum
    print 'Measuring and subtracting the continuum...'
    t0 = time.time()
    cube_cont_remove, cont_params = remove_cont(cube, exclude=cont_exclude)
    t1 = time.time()
    print 'Continuum successfully subtracted in', t1-t0, 'seconds.'

    # Determine the RMS around the line
    print 'Measuring RMS across the cube...'
    local_rms = calc_local_rms(cube_cont_remove, exclude=cont_exclude)
    print 'RMS measurement complete!'

    # Create a mask of pixels to skip in the fitting
    print 'Creating a mask to skip model fitting for spaxels with S/N <', sn_thresh
    skippix = skip_pixels(cube_cont_remove, local_rms, sn_thresh=sn_thresh, spec_use=guess_region)

    print 'Fitting the cube...'
    t2 = time.time()
    results = cubefit(cube_cont_remove, model, skip=skippix, line_centers=line_centers, exclude=fit_exclude,
                                 auto_guess=auto_guess, guess_type=guess_type, guess_region=guess_region, calc_uncert=calc_uncert,
                                 rms=local_rms, nmc=nmc, cores=cores, parallel=parallel)
    t3 = time.time()
    print 'Cube successfully fit in', t3-t2, 'seconds.'
    fit_params = results[0]
    resids = results[1]

    if calc_uncert:
        fit_params_mc = results[2]

        total_results = {'continuum_sub': cube_cont_remove,
                         'cont_params': cont_params,
                         'fit_params': fit_params,
                         'fit_params_mc': fit_params_mc,
                         'fit_pixels': skippix,
                         'residuals': resids,
                         'rms': local_rms}
    else:
        total_results = {'continuum_sub': cube_cont_remove,
                         'cont_params': cont_params,
                         'fit_params': fit_params,
                         'fit_pixels': skippix,
                         'residuals': resids,
                         'rms': local_rms}

    return total_results

def write_files(results, header, savedir='', suffix=''):
    """
    Writes out all of the results to FITS files.
    """

    key_remove = ['CDELT3', 'CRPIX3', 'CUNIT3', 'CTYPE3', 'CRVAL3']

    # Write out the continuum-subtracted spectral cube and the residuals
    results['continuum_sub'].write(savedir+'continuum_sub'+suffix+'.fits', format='fits', overwrite=True)
    results['residuals'].write(savedir+'residuals'+suffix+'.fits', format='fits', overwrite=True)

    # Write out the best parameters for the continuum
    hdu_cont_params =fits.PrimaryHDU(data=results['cont_params'], header=header)
    hdu_cont_params.header.remove('WCSAXES')
    for k in key_remove:
        hdu_cont_params.header.remove(k)
    hdu_cont_params.header.remove('BUNIT')
    fits.HDUList([hdu_cont_params]).writeto(savedir+'cont_params'+suffix+'.fits', overwrite=True)

    # Write out the pixels that were fit or skipped
    hdu_skip =fits.PrimaryHDU(data=np.array(results['fit_pixels'], dtype=int), header=header)
    hdu_skip.header['WCSAXES'] = 2
    for k in key_remove:
        hdu_skip.header.remove(k)
    hdu_skip.header.remove('BUNIT')
    fits.HDUList([hdu_skip]).writeto(savedir+'skippix'+suffix+'.fits', overwrite=True)

    # Write out the rms estimate
    hdu_rms =fits.PrimaryHDU(data=np.array(results['rms'].value), header=header)
    hdu_rms.header['WCSAXES'] = 2
    for k in key_remove:
        hdu_rms.header.remove(k)
    fits.HDUList([hdu_rms]).writeto(savedir+'rms'+suffix+'.fits', overwrite=True)

    # For each line fit, write out both the best fit gaussian parameters
    # and physical line parameters
    lines = results['fit_params'].keys()

    # See if an uncertainty estimate was made
    unc_exist = results.has_key('fit_params_mc')

    for l in lines:

        gauss_params = results['fit_params'][l]
        hdu_amp = fits.PrimaryHDU(data=gauss_params['amplitude'].value, header=header)
        hdu_cent = fits.ImageHDU(data=gauss_params['mean'].value, header=header)
        hdu_sig = fits.ImageHDU(data=gauss_params['sigma'].value, header=header)

        line_params = results['line_params'][l]
        hdu_flux = fits.PrimaryHDU(data=line_params['int_flux'].value, header=header)
        hdu_vel = fits.ImageHDU(data=line_params['velocity'].value, header=header)
        hdu_vdisp = fits.ImageHDU(data=line_params['veldisp'].value, header=header)

        if unc_exist:
            gauss_params_mc = results['fit_params_mc'][l]
            hdu_amp_mc = fits.PrimaryHDU(data=gauss_params_mc['amplitude'].value, header=header)
            hdu_cent_mc = fits.ImageHDU(data=gauss_params_mc['mean'].value, header=header)
            hdu_sig_mc = fits.ImageHDU(data=gauss_params_mc['sigma'].value, header=header)

            hdu_flux_err = fits.ImageHDU(data=line_params['int_flux_err'].value, header=header)
            hdu_vel_err = fits.ImageHDU(data=line_params['velocity_err'].value, header=header)
            hdu_vdisp_err = fits.ImageHDU(data=line_params['veldisp_err'].value, header=header)

        hdu_amp.header['EXTNAME'] = 'amplitude'
        hdu_cent.header['EXTNAME'] = 'line center'
        hdu_sig.header['EXTNAME'] = 'sigma'

        hdu_flux.header['EXTNAME'] = 'int flux'
        hdu_vel.header['EXTNAME'] = 'velocity'
        hdu_vdisp.header['EXTNAME'] = 'velocity dispersion'

        hdu_amp.header['WCSAXES'] = 2
        hdu_cent.header['WCSAXES'] = 2
        hdu_sig.header['WCSAXES'] = 2

        hdu_flux.header['WCSAXES'] = 2
        hdu_vel.header['WCSAXES'] = 2
        hdu_vdisp.header['WCSAXES'] = 2

        if unc_exist:
            hdu_amp_mc.header['EXTNAME'] = 'MC amplitude'
            hdu_cent_mc.header['EXTNAME'] = 'MC line center'
            hdu_sig_mc.header['EXTNAME'] = 'MC sigma'

            hdu_flux_err.header['EXTNAME'] = 'int flux error'
            hdu_vel_err.header['EXTNAME'] = 'velocity error'
            hdu_vdisp_err.header['EXTNAME'] = 'velocity dispersion error'

            hdu_amp.header['WCSAXES'] = 2
            hdu_cent.header['WCSAXES'] = 2
            hdu_sig.header['WCSAXES'] = 2

            hdu_flux.header['WCSAXES'] = 2
            hdu_vel.header['WCSAXES'] = 2
            hdu_vdisp.header['WCSAXES'] = 2

        for k in key_remove:
            hdu_amp.header.remove(k)
            hdu_cent.header.remove(k)
            hdu_sig.header.remove(k)
            hdu_flux.header.remove(k)
            hdu_vel.header.remove(k)
            hdu_vdisp.header.remove(k)

            if unc_exist:
                hdu_amp_mc.header.remove(k)
                hdu_cent_mc.header.remove(k)
                hdu_sig_mc.header.remove(k)
                hdu_flux_err.header.remove(k)
                hdu_vel_err.header.remove(k)
                hdu_vdisp_err.header.remove(k)

        hdu_cent.header['BUNIT'] = 'micron'
        hdu_sig.header['BUNIT'] = 'micron'
        hdu_flux.header['BUNIT'] = 'W m-2'
        hdu_vel.header['BUNIT'] = 'km s-1'
        hdu_vdisp.header['BUNIT'] = 'km s-1'

        if unc_exist:
            hdu_cent_mc.header['BUNIT'] = 'micron'
            hdu_sig_mc.header['BUNIT'] = 'micron'
            hdu_flux_err.header['BUNIT'] = 'W m-2'
            hdu_vel_err.header['BUNIT'] = 'km s-1'
            hdu_vdisp_err.header['BUNIT'] = 'km s-1'

        gauss_list = fits.HDUList([hdu_amp, hdu_cent, hdu_sig])
        gauss_list.writeto(savedir+l+'_gauss_params'+suffix+'.fits', overwrite=True)
        if unc_exist:
            gauss_mc_list = fits.HDUList([hdu_amp_mc, hdu_cent_mc, hdu_sig_mc])
            gauss_mc_list.writeto(savedir+l+'_gauss_params_mc'+suffix+'.fits', overwrite=True)
            line_list = fits.HDUList([hdu_flux, hdu_vel, hdu_vdisp, hdu_flux_err, hdu_vel_err, hdu_vdisp_err])
            line_list.writeto(savedir+l+'_line_params'+suffix+'.fits', overwrite=True)
        else:
            line_list = fits.HDUList([hdu_flux, hdu_vel, hdu_vdisp])
            line_list.writeto(savedir+l+'_line_params'+suffix+'.fits', overwrite=True)

    return 0


def plot_specfit(cube, model, fit_params, pixel, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    y, x = pixel
    spec = cube[:, x, y]
    waves = spec.spectral_axis.to(u.micron).value
    flux = spec.value
    bestfit_model = model.copy()
    nlines = len(fit_params.keys())

    for k in fit_params.keys():

        if (nlines == 1):

            bestfit_model.amplitude = fit_params[k]['amplitude'][x, y].value
            bestfit_model.mean = fit_params[k]['mean'][x, y].value
            bestfit_model.stddev = fit_params[k]['sigma'][x, y].value

        else:

            bestfit_model[k].amplitude = fit_params[k]['amplitude'][x, y].value
            bestfit_model[k].mean = fit_params[k]['mean'][x, y].value
            bestfit_model[k].stddev = fit_params[k]['sigma'][x, y].value

    ax.plot(waves, flux, ls='steps-mid', color='k', lw=0.5)
    ax.plot(waves, bestfit_model(waves), color='r', lw=1.0)
    ax.set_xlim((np.min(waves), np.max(waves)))
    ax.set_xlabel('Wavelength [micron]')
    ax.set_ylabel('Flux ['+spec.unit.to_string()+']')

    return fig


def read_line_params(file):
    """
    Function to read a FITS file containing the physical line params and return a
    dictionary with the same structure as what is output of calc_line_params.
    """

    hdu_list = fits.open(file)

    line_params = {}
    line_params['int_flux'] = hdu_list['int flux'].data*u.W/u.m**2/u.micron
    line_params['velocity'] = hdu_list['velocity'].data*u.km/u.s
    line_params['veldisp'] = hdu_list['velocity dispersion'].data*u.km/u.s

    if len(hdu_list) == 6:
        line_params['int_flux_err'] = hdu_list['int flux error'].data*u.W/u.m**2/u.micron
        line_params['velocity_err'] = hdu_list['velocity error'].data*u.km/u.s
        line_params['veldisp_err'] = hdu_list['velocity dispersion error'].data*u.km/u.s

    return line_params


def read_fit_params(files, line_names):
    """
    Function to read the FITS file(s) for the fit parameters of a spectral model.
    """

    if type(line_names) == str:
        line_names = [line_names]
        files = [files]

    fit_params = {n:{'amplitude':[], 'mean':[], 'sigma':[]} for n in line_names}

    for i, f in enumerate(files):

        hdu_list = fits.open(f)
        fit_params[line_names[i]]['amplitude'] = hdu_list[0].data*u.W/u.m**2/u.micron
        fit_params[line_names[i]]['mean'] = hdu_list[1].data*u.micron
        fit_params[line_names[i]]['sigma'] = hdu_list[2].data*u.micron

    return fit_params


def fitpa(vel, vel_err=None, xoff=None, yoff=None, mask=None):
    """
    Function to fit the kinematic position angle using fit_kinematic_pa from Michele Cappelari
    """

    from fit_kinematic_pa import fit_kinematic_pa

    x, y = np.meshgrid(range(vel.shape[1]), range(vel.shape[0]))

    if vel_err is None:
        vel_err = vel*0 + 10.

    if mask is None:
        mask = np.isnan(vel)

    xx = x.flatten()
    yy = y.flatten()
    vel_flat = vel.flatten()
    vel_err_flat = vel_err.flatten()
    mask_flat = mask.flatten()

    xx = xx[~mask_flat]
    yy = yy[~mask_flat]
    vel_flat = vel_flat[~mask_flat]
    vel_err_flat = vel_err_flat[~mask_flat]

    if xoff is None:
        xoff = vel.shape[0]/2
    if yoff is None:
        yoff = vel.shape[1]/2

    xx = xx-xoff
    yy = yy-yoff

    angBest, angErr, vSyst = fit_kinematic_pa(xx, yy, vel_flat, dvel=vel_err_flat)

    return angBest, angErr, vSyst


def find_cont_center(cube, lamrange):
    """
    Function to fit a 2D Gaussian to the image of a user-defined continuum
    """

    slice = cube.spectral_slab(lamrange[0], lamrange[1])
    int = slice.moment(0)   # Zeroth moment is just the integral along the spectrum
    img = int.value/np.nanmean(int.value)
    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    guess_x = img.shape[1]/2
    guess_y = img.shape[0]/2
    img_cut = img[guess_y-15:guess_y+15, guess_x-15:guess_x+15]
    xx_cut = xx[guess_y-15:guess_y+15, guess_x-15:guess_x+15]
    yy_cut = yy[guess_y-15:guess_y+15, guess_x-15:guess_x+15]
    gauss_mod = apy_mod.models.Gaussian2D(x_mean=guess_x, y_mean=guess_y,
                                          x_stddev=3.0, y_stddev=3.0)
    fitter = apy_mod.fitting.LevMarLSQFitter()

    best_fit = fitter(gauss_mod, xx_cut, yy_cut, img_cut)

    center = [best_fit.x_mean.value, best_fit.y_mean.value]

    return center


def findpeaks(spec, lam, model, guess_region, line_centers):
    """
    Function to find the peaks in a spectral region by smoothing with a Gaussian and
    then using a peak finding algorithm. Then use the found peaks to adjust the initial
    guesses for the model.
    """

    # Make a copy of the model
    mod = model.copy()

    # Determine the number of lines being fit within the model
    nlines = len(line_centers.keys())

    # Smooth the spectrum with a 2 pixel wide Gaussian kernel to get rid of high frequency noise
    gauss_kern = apy_conv.Gaussian1DKernel(2.0)
    smoothed_spec = apy_conv.convolve(spec,gauss_kern)

    # Find the peaks with peakutils
    ind_peaks = peakutils.indexes(smoothed_spec[guess_region])
    peak_waves = lam[guess_region][ind_peaks]
    peak_flux = spec[guess_region][ind_peaks]
    print ind_peaks
    # Sort the peaks by flux and take the top N as estimates for the lines in the model
    ind_sort = np.argsort(peak_flux)
    peak_waves_sort = peak_waves[ind_sort]
    peak_flux_sort = peak_flux[ind_sort]

    if nlines > 1:
        lc = np.array([line_centers[k].value for k in line_centers.keys()])
        ln = np.array([k for k in line_centers.keys()])

        # Sort the lines being fit by wavelength
        ind_sort_lc = np.argsort(lc)
        lc_sort = lc[ind_sort_lc]
        ln_sort = ln[ind_sort_lc]

        if len(ind_peaks) >= nlines:

            peak_waves_use = peak_waves_sort[0:nlines]
            peak_flux_use = peak_flux_sort[0:nlines]

            # Now sort the useable peaks by wavelength and associate with the modeled lines
            ind_sort_use = np.argsort(peak_waves_use)
            peak_waves_use_sort = peak_waves_use[ind_sort_use]
            peak_flux_use_sort = peak_flux_use[ind_sort_use]

            for g, n in enumerate(ln_sort):

                mod[n].mean = peak_waves_use_sort[g]
                mod[n].amplitude = peak_flux_use_sort[g]

        elif len(ind_peaks) > 0:

            # Need to figure out which line each peak is associated with
            # Use the difference between the peak wavelength and the line center
            # Whichever line is closest to the peak is the one we'll assume the peak
            #    is associated with.

            for l, w in enumerate(peak_waves_sort):

                closest_line = ln[np.argmin(np.abs(lc-w))]
                mod[closest_line].mean = w
                mod[closest_line].amplitude = peak_flux_sort[l]

    else:

        # If there is only one line then just take the strongest peak
        mod.mean = peak_waves_sort[0]
        mod.amplitude = peak_flux_sort[0]

    return mod
