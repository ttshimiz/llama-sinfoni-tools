# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:56:56 2016

@author: ttshimiz
"""

import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import astropy.modeling as apy_mod
from astropy.stats import sigma_clipped_stats, sigma_clip
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


def calc_line_params(fit_params, line_centers, inst_broad=0):
    """
    Function to determine the integrated line flux, velocity, and linewidth
    Assumes the units on the amplitude are W/m^2/micron and the units on the
    mean and sigma are micron as well.
    Also determines the S/N of the line using the local rms of the spectrum.
    """

    line_params = {}

    for i,k in enumerate(fit_params.keys()):

        lc = line_centers[i]
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

    ax_int.show_colorscale(cmap='cubehelix', stretch='arcsinh', vmin=0, vmid=-np.nanmax(int_flux_hdu.data)/1000.)
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
    if type(line_centers) == u.quantity.Quantity:
        line_centers = [line_centers]
        line_names = [line_names]
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

def cubefit(cube, model, skip=None, exclude=None, max_guess=False, guess_region=None,
            calc_uncert=False, nmc=1000., rms=None):
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

    if calc_uncert:
        fit_params_err = {}

    if hasattr(model, 'submodel_names'):

        for n in model.submodel_names:
            fit_params[n] = {'amplitude': np.zeros((xsize, ysize))*flux_unit*np.nan,
                             'mean': np.zeros((xsize, ysize))*spec_ax_unit*np.nan,
                             'sigma': np.zeros((xsize, ysize))*spec_ax_unit*np.nan}
            if calc_uncert:
               fit_params_err[n] = {'amp_err': np.zeros((nmc, xsize, ysize))*flux_unit*np.nan,
                             'mean_err': np.zeros((nmc, xsize, ysize))*spec_ax_unit*np.nan,
                             'sigma_err': np.zeros((nmc, xsize, ysize))*spec_ax_unit*np.nan}

    else:
        fit_params[model.name] = {'amplitude': np.zeros((xsize, ysize))*flux_unit*np.nan,
                                  'mean': np.zeros((xsize, ysize))*spec_ax_unit*np.nan,
                                  'sigma': np.zeros((xsize, ysize))*spec_ax_unit*np.nan}
        if calc_uncert:
           fit_params_err[model.name] = {'amp_err': np.zeros((nmc, xsize, ysize))*flux_unit*np.nan,
                                         'mean_err': np.zeros((nmc, xsize, ysize))*spec_ax_unit*np.nan,
                                         'sigma_err': np.zeros((nmc, xsize, ysize))*spec_ax_unit*np.nan}

    if skip is None:
        skip = np.zeros((xsize, ysize), dtype=np.bool)

    for i in range(xsize):
        for j in range(ysize):

            spec = cube[:, i, j].value/10**(-17)
            if calc_uncert:
                rms_i = rms[i, j].value/10**(-17)
            else:
                rms_i = None

            if (np.any(~np.isnan(spec)) & ~skip[i, j]):

                if max_guess:
                    if guess_region is None:
                         guess_region = np.ones(len(spec), dtype=np.bool)
                    ind_max = np.argmax(spec[guess_region])
                    wave_max = lam[guess_region][ind_max]
                    flux_max = spec[guess_region][ind_max]

                    if hasattr(model, 'submodel_names'):
                        model.amplitude_0 = flux_max
                        model.mean_0 = wave_max
                    else:
                        model.amplitude = flux_max
                        model.mean = wave_max

                fit_results = specfit(lam, spec, model, exclude=exclude, calc_uncert=calc_uncert, nmc=nmc, rms=rms_i)

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

                            fit_params_err[n]['amp_err'][:,i,j] = mc_amps*flux_unit*10**(-17)
                            fit_params_err[n]['mean_err'][:,i,j] = mc_mean*spec_ax_unit
                            fit_params_err[n]['sigma_err'][:,i,j] = mc_sig*spec_ax_unit
                else:
                    fit_params[model.name]['amplitude'][i,j] = best_fit.amplitude.value*flux_unit*10**(-17)
                    fit_params[model.name]['mean'][i,j] = best_fit.mean.value*spec_ax_unit
                    fit_params[model.name]['sigma'][i,j] = best_fit.stddev.value*spec_ax_unit

                    if calc_uncert:
                        mc_amps = np.array([err_fits[k].amplitude.value for k in range(nmc)])
                        mc_mean = np.array([err_fits[k].mean.value for k in range(nmc)])
                        mc_sig = np.array([err_fits[k].stddev.value for k in range(nmc)])

                        fit_params_err[model.name]['amp_err'][:,i,j] = mc_amps*flux_unit*10**(-17)
                        fit_params_err[model.name]['mean_err'][:,i,j] = mc_mean*spec_ax_unit
                        fit_params_err[model.name]['sigma_err'][:,i,j] = mc_sig*spec_ax_unit

                residuals[:,i,j] = (spec - best_fit(spec_ax.to(u.micron).value))*10**(-17)
            else:
                residuals[:,i,j] = spec*10**(-17)

    resid_cube = SpectralCube(data=residuals, wcs=cube.wcs,
                              meta={'BUNIT':cube.unit.to_string()})
    resid_cube = resid_cube.with_spectral_unit(cube.spectral_axis.unit)

    if calc_uncert:
        return fit_params, resid_cube, fit_params_err
    else:
        return fit_params, resid_cube


def specfit(x, fx, model, exclude=None, calc_uncert=False, rms=None, nmc=100):
    """
    Function to fit a single spectrum with a model.
    """

    if exclude is not None:
        x = x[~exclude]
        fx = fx[~exclude]

    fitter = apy_mod.fitting.LevMarLSQFitter()
    bestfit = fitter(model, x, fx)

    if calc_uncert:
        rand_fits = []
        for i in range(nmc):
            rand_spec = np.random.randn(len(fx))*rms + fx
            rand_fit_i = fitter(model, x, rand_spec)
            rand_fits.append(rand_fit_i)

        return [bestfit, rand_fits]
    else:
        return bestfit


def skip_pixels(cube, rms, sn_thresh=3.0, exclude=None):
    """
    Function to determine which pixels to skip based on a user defined S/N threshold.
    Returns an NxM boolean array where True indicates a pixel to skip.
    The signal used is the maximum value in the spectrum.
    If the maximum value is a NaN then that pixel is also skipped.
    """

    if exclude is None:
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
                s_n = np.max(s[~exclude])/rms[x,y]
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


def runfit(cube, model, sn_thresh=3.0, cont_exclude=None, fit_exclude=None,
           max_guess=False, guess_region=None):

    # Subtract out the continuum
    cube_cont_remove, cont_params = remove_cont(cube, exclude=cont_exclude)

    # Determine the RMS around the line
    local_rms = calc_local_rms(cube_cont_remove, exclude=cont_exclude)

    # Create a mask of pixels to skip in the fitting
    skippix = skip_pixels(cube_cont_remove, local_rms, sn_thresh=sn_thresh, exclude=fit_exclude)

    fit_params, resids = cubefit(cube_cont_remove, model, skip=skippix, exclude=fit_exclude,
                                 max_guess=max_guess, guess_region=guess_region)

    results = {'continuum_sub': cube_cont_remove,
               'cont_params': cont_params,
               'fit_params': fit_params,
               'fit_pixels': skippix,
               'residuals': resids}

    return results

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
    fits.HDUList([hdu_cont_params]).writeto(savedir+'cont_params'+suffix+'.fits', clobber=True)

     # Write out the pixels that were fit or skipped
    hdu_skip =fits.PrimaryHDU(data=np.array(results['fit_pixels'], dtype=int), header=header)
    hdu_skip.header['WCSAXES'] = 2
    for k in key_remove:
        hdu_skip.header.remove(k)
    hdu_skip.header.remove('BUNIT')
    fits.HDUList([hdu_skip]).writeto(savedir+'skippix'+suffix+'.fits', clobber=True)

    # For each line fit, write out both the best fit gaussian parameters
    # and physical line parameters
    lines = results['fit_params'].keys()

    for l in lines:

        gauss_params = results['fit_params'][l]
        hdu_amp = fits.PrimaryHDU(data=gauss_params['amplitude'].value, header=header)
        hdu_cent = fits.ImageHDU(data=gauss_params['mean'].value, header=header)
        hdu_sig = fits.ImageHDU(data=gauss_params['sigma'].value, header=header)

        line_params = results['line_params'][l]
        hdu_flux = fits.PrimaryHDU(data=line_params['int_flux'].value, header=header)
        hdu_vel = fits.ImageHDU(data=line_params['velocity'].value, header=header)
        hdu_vdisp = fits.ImageHDU(data=line_params['veldisp'].value, header=header)

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

        for k in key_remove:
            hdu_amp.header.remove(k)
            hdu_cent.header.remove(k)
            hdu_sig.header.remove(k)
            hdu_flux.header.remove(k)
            hdu_vel.header.remove(k)
            hdu_vdisp.header.remove(k)

        hdu_cent.header['BUNIT'] = 'micron'
        hdu_sig.header['BUNIT'] = 'micron'
        hdu_flux.header['BUNIT'] = 'W m-2'
        hdu_vel.header['BUNIT'] = 'km s-1'
        hdu_vdisp.header['BUNIT'] = 'km s-1'

        gauss_list = fits.HDUList([hdu_amp, hdu_cent, hdu_sig])
        gauss_list.writeto(savedir+l+'_gauss_params'+suffix+'.fits', clobber=True)
        line_list = fits.HDUList([hdu_flux, hdu_vel, hdu_vdisp])
        line_list.writeto(savedir+l+'_line_params'+suffix+'.fits', clobber=True)

    return 0


