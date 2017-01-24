# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 18:17:01 2016

@author: ttshimiz

File containing all of the relevant NIR emission and absorption lines
in our SINFONI H+K data.

All wavelengths are vacuum in microns.

Molecular hydrogen lines from Wolfire & Koenig 1991.
Atomic lines from http://www.pa.uky.edu/~peter/atomic/
CO absorption lines from http://www.gemini.edu/sciops/instruments/nir/wavecal/colines.dat

"""

import astropy.units as u

EMISSION_LINES = {'H2 (1-0) S(0)': 2.223*u.micron,
                  'H2 (1-0) S(1)': 2.1218*u.micron,
                  'H2 (1-0) S(2)': 2.033*u.micron,
                  'H2 (2-1) S(1)': 2.247*u.micron,
                  'H2 (2-1) S(3)': 2.073*u.micron,
                  'H2 (1-0) S(3)': 1.957*u.micron,
                  'H2 (2-1) S(0)': 2.355*u.micron,
                  'H2 (2-1) S(2)': 2.154*u.micron,
                  'H2 (3-2) S(2)': 2.286*u.micron,
                  'H2 (3-2) S(3)': 2.201*u.micron,
                  '[SiVI]': 1.965*u.micron,
                  '[AlIX]': 2.04*u.micron,
                  '[CaVIII]': 2.3211*u.micron,
                  '[SiXI]': 1.936*u.micron,
                  'Bry': 2.16612*u.micron,
                  'HeI': 2.058*u.micron,
                  'Brd': 1.9451*u.micron,
                  'Paa': 1.8756*u.micron,
                  'Br11': 1.6811*u.micron,
                  'Br10': 1.7367*u.micron,
                  '[FeII]': 1.644*u.micron}

ABSORPTION_LINES = {'12CO (3-0)': 1.5582*u.micron,
                    '12CO (4-1)': 1.5780*u.micron,
                    '12CO (5-2)': 1.5982*u.micron,
                    '12CO (6-3)': 1.6187*u.micron,
                    '12CO (7-4)': 1.6397*u.micron,
                    '12CO (8-5)': 1.6610*u.micron,
                    '12CO (9-6)': 1.68*u.micron,
                    '12CO (10-7)': 1.71*u.micron,
                    'SiI': 1.58759219*u.micron,
                    'NaI': 2.208*u.micron,
                    'CaI': 2.26*u.micron,
                    '12CO (2-0)': 2.2935*u.micron,
                    '12CO (3-1)': 2.3227*u.micron,
                    '12CO (4-2)': 2.3525*u.micron,
                    '12CO (5-3)': 2.3829*u.micron}
