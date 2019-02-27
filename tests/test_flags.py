# TODO
def default_flags():
    assert map_default_double.y.flags.writeable == False
    assert map_default_double[:, :].flags.writeable == False
    assert map_default_double.u.flags.writeable == False
    assert map_default_double[:].flags.writeable == False


def spectral_flags():
    assert map_spectral_double.y.flags.writeable == False
    assert map_spectral_double[:, :].flags.writeable == False
    assert map_spectral_double[:, :][0].flags.writeable == False
    assert map_spectral_double.u.flags.writeable == False
    assert map_spectral_double[:].flags.writeable == False
    assert map_spectral_double[:][0].flags.writeable == False