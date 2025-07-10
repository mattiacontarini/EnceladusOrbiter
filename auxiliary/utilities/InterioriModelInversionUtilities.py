def tidal_response(Interior_Model, Numerics, Forcing, eng=None):
    """
    Compute the tidal response of the moon.
    Reference: Rovira-Navarro 2024.

    Parameters
    ----------
    Interior_Model : list of dict
        List of dictionaries containing the parameters of the moon's interior.
    Numerics : dict
        Dictionary containing the numerical settings.
    Forcing : list of dict
        List of dictionaries containing the forcing parameters.

    Returns
    -------
    k2 : complex
        Gravitational Love number of degree 2.
    h2 : complex
        radial displacement Love number of degree 2.
    """

    quit = False
    if eng is None:
        eng = love3d.initialize()
        quit = True

    LoveSpectra, y = eng.compute_Love(Interior_Model, Numerics, Forcing, nargout=2)
    k2 = LoveSpectra["k"]
    h2 = LoveSpectra["h"]

    if quit:
        eng.quit()

    return k2, h2
