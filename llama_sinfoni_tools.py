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
    cont_fit = fitter(cont, x, spectrum, weights=1./errors)
    
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
    
    
def gauss_fit_single(x, spectrum, guess=None, errors=None, exclude=None):
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
    
    
    fitter = apy_mod.fitting.LevMarLSQFitter()     
    gauss_fit = fitter(model, x, spectrum, weights=1./errors)
    
    return gauss_fit
        
def cubefit_gauss(cube, guess=None, exclude=None):
    """
    Function to loop through all of the spectra in a cube and fit a gaussian
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
            
            if np.any(~np.isnan(spec)):
                gauss_mod = gauss_fit_single(spec_ax, spec, guess=guess, exclude=exclude)
            
                fit_params['amplitude'][i, j] = gauss_mod.amplitude.value*flux_unit
                fit_params['mean'][i, j] = gauss_mod.mean.value*spec_ax_unit
                fit_params['sigma'][i, j] = gauss_mod.stddev.value*spec_ax_unit
                
            else:
				fit_params['amplitude'][i, j] = np.nan
				fit_params['mean'][i, j] = np.nan
				fit_params['sigma'][i, j] = np.nan
        
    return fit_params 
    

def calc_line_params(fit_params, line_center):
    """
    Function to determine the integrated line flux, velocity, and linewidth
    Assumes the units on the amplitude are W/m^2/micron and the units on the
    mean and sigma are micron as well.
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
    line_params['veldisp'] = veldisp
    
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
    
    int_mn, int_med, int_sig = sigma_clipped_stats(line_params['int_flux'].value)
    vel_mn, vel_med, vel_sig = sigma_clipped_stats(line_params['velocity'].value)
    vdp_mn, vdp_med, vdp_sig = sigma_clipped_stats(line_params['veldisp'].value)
    
    ax_int.show_colorscale(cmap='cubehelix', vmin=int_med-2*int_sig, vmax=int_med+2*int_sig)
    ax_vel.show_colorscale(cmap='RdBu_r', vmin=vel_med-2*vel_sig, vmax=vel_med+2*vel_sig)
    ax_vdp.show_colorscale(cmap='Spectral', vmin=vdp_med-2*vdp_sig, vmax=vdp_med+2*vdp_sig)
    
    ax_int.show_colorbar()
    ax_vel.show_colorbar()
    ax_vdp.show_colorbar()
    
    ax_int.colorbar.set_axis_label_text(r'Flux 10$^{-17}$ [W m$^{-2}$]')
    ax_vel.colorbar.set_axis_label_text(r'Velocity [km s$^{-1}$]')
    ax_vdp.colorbar.set_axis_label_text(r'$\sigma_{\rm v}$ [km s$^{-1}$]')
    
    ax_int.set_axis_labels_ydisp(-30)
    ax_vel.hide_yaxis_label()
    ax_vel.hide_ytick_labels()
    ax_vdp.hide_yaxis_label()
    ax_vdp.hide_ytick_labels()
    
    fig.subplots_adjust(wspace=0.3)
    
    return fig, [ax_int, ax_vel, ax_vdp]
    
    
def run_line(cube, line_name, velrange =[-4000, 4000],
              zz=0, plot_results=True):
    
    # Get the rest wavelength          
    line_center = lines.EMISSION_LINES[line_name]*(1+zz)   
    
    # Slice the cube
    slice = cube.with_spectral_unit(unit=u.km/u.s, velocity_convention='optical',
                                    rest_value=line_center).spectral_slab(velrange[0]*u.km/u.s, velrange[1]*u.km/u.s) 
    slice = slice.with_spectral_unit(unit=u.micron, velocity_convention='optical',
                                     rest_value=line_center)
    
    # Subtract out the continuum
    cube_cont_remove, cont_params = remove_cont(slice)
    
    # Fit a Gaussian to the line
    gaussfit_params = cubefit_gauss(cube_cont_remove)
    
    # Calculate the line parameters
    line_params = calc_line_params(gaussfit_params, line_center)
    
    results = {'line_params': line_params,
               'continuum_sub': cube_cont_remove,
               'gauss_params': gaussfit_params,
               'data': slice}
    
    if plot_results:
        fig, axes = plot_line_params(line_params, slice.header)
        results['results_fig'] = fig
        results['results_axes'] = axes
    
    return results
    