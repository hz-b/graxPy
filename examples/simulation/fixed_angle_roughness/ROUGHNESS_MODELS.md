# How the two roughness models are computed

**Debye-Waller** applies a scalar damping factor to the diffraction
efficiencies, without touching the grating geometry:

    damping = exp(-(4*pi*sigma*sin(theta)/wavelength)^2)

`sigma` is the rms roughness height and `theta` the grazing incidence
angle. It approximates the loss from unresolved, sub-wavelength
roughness and has no lateral correlation length of its own.

**Random-interface** roughness instead perturbs the actual interface
shape with a periodic, zero-mean Gaussian random field sampled per
grating period. Its lateral correlation is set via a Gaussian power
spectrum:

    amplitude(k) = exp(-0.25 * (k * xi)^2)

where `k` are the spatial frequencies over one period and `xi` is
`correlation_length_nm`. The field is built in Fourier space, inverse
transformed, then rescaled to the target rms `sigma`. Larger `xi` means
smoother, longer-wavelength height variations; `xi = 0` gives white
noise (fully uncorrelated); `xi = None` defaults to `period/10`.
