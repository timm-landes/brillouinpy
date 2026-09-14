import numpy as np

from conftest import require_optional
from brillouinpy.core import SpectralImage

ramanspy = require_optional("ramanspy")


def _dho(x, amplitude, freq_shift, linewidth, background=0.0, axis_shift=0.0):
    xs = x - axis_shift
    return amplitude * 4 * linewidth * freq_shift ** 2 / (
        np.pi * ((xs ** 2 - freq_shift ** 2) ** 2 + 4 * (linewidth * xs) ** 2)
    ) + background


def test_ramanspy_cropper_accepts_brillouinpy_spectral_object():
    # The design doc (docs/design/multimodal_container.md) claims a brillouinpy
    # SpectralObject can be passed straight into RamanSPy operations, since
    # `channels` is purely additive and the `spectral_data`/`spectral_axis`
    # attributes RamanSPy expects are untouched. ramanspy.preprocessing.misc.Cropper
    # is a representative, deterministic RamanSPy operation: it deep-copies the
    # object handed to it and only ever touches those two attributes, with no
    # isinstance check on the input type.
    axis = np.linspace(-20, 20, 60)
    data = np.stack([
        [_dho(axis, 5e-3, 8.5, 1.0) for _ in range(3)]
        for _ in range(4)
    ])
    image = SpectralImage(data, axis)

    cropper = ramanspy.preprocessing.misc.Cropper(region=(-10, 10))
    cropped = cropper.apply(image)

    in_region = (axis >= -10) & (axis <= 10)

    # RamanSPy's PreprocessingStep deep-copies the object it's given rather than
    # constructing a new one of its own type, so the brillouinpy class survives.
    assert type(cropped) is SpectralImage
    assert cropped is not image
    assert cropped.shape == image.shape
    assert cropped.spectral_data.shape == (4, 3, int(in_region.sum()))
    assert np.allclose(cropped.spectral_axis, axis[in_region])
    assert np.allclose(
        np.ma.filled(cropped.spectral_data, np.nan),
        np.ma.filled(image.spectral_data[..., in_region], np.nan),
        equal_nan=True,
    )
    # the original object is untouched
    assert image.spectral_data.shape == (4, 3, 60)


def _image_with_raman_channel():
    axis = np.linspace(-20, 20, 60)
    data = np.stack([
        [_dho(axis, 5e-3, 8.5, 1.0) for _ in range(3)]
        for _ in range(4)
    ])
    raman_axis = np.linspace(500, 3000, 40)
    raman_data = np.stack([
        [_dho(raman_axis, 1.0, 1800, 50.0) for _ in range(3)]
        for _ in range(4)
    ])
    raman = SpectralImage(raman_data, raman_axis)
    return SpectralImage(data, axis, channels={"raman": raman}), raman


def test_ramanspy_step_applies_directly_to_a_channel_value():
    # A channel value is validated (core._validate_channels) to be a full
    # SpectralObject, not a bare array - so it carries the same untouched
    # spectral_data/spectral_axis attributes as any top-level object, and the
    # same duck-typing compatibility applies to it directly.
    image, raman = _image_with_raman_channel()

    cropper = ramanspy.preprocessing.misc.Cropper(region=(1000, 2000))
    cropped = cropper.apply(image.channels["raman"])

    in_region = (raman.spectral_axis >= 1000) & (raman.spectral_axis <= 2000)

    assert type(cropped) is SpectralImage
    assert cropped.spectral_data.shape == (4, 3, int(in_region.sum()))
    assert np.allclose(cropped.spectral_axis, raman.spectral_axis[in_region])
    assert np.allclose(
        np.ma.filled(cropped.spectral_data, np.nan),
        np.ma.filled(raman.spectral_data[..., in_region], np.nan),
        equal_nan=True,
    )
    # the channel on the original container is untouched
    assert image.channels["raman"].spectral_data.shape == (4, 3, 40)


def test_apply_to_channel_with_real_ramanspy_step():
    # SpectralContainer.apply_to_channel(name, step) only ever calls
    # step.apply(self.channels[name]), with no isinstance check on `step` - so a
    # real RamanSPy PreprocessingStep is a drop-in `step` for it.
    image, raman = _image_with_raman_channel()

    cropper = ramanspy.preprocessing.misc.Cropper(region=(1000, 2000))
    transformed = image.apply_to_channel("raman", cropper)

    in_region = (raman.spectral_axis >= 1000) & (raman.spectral_axis <= 2000)

    assert transformed.channels["raman"].spectral_data.shape == (4, 3, int(in_region.sum()))
    assert np.allclose(transformed.channels["raman"].spectral_axis, raman.spectral_axis[in_region])
    # the original container's channel is untouched
    assert image.channels["raman"].spectral_data.shape == (4, 3, 40)
    assert np.allclose(image.channels["raman"].spectral_axis, raman.spectral_axis)


def test_ramanspy_arrays_into_brillouinpy_container_then_spatial_op():
    # Cheap regression guard for the reverse direction: arrays coming out of a
    # RamanSPy SpectralImage are plain ndarrays that brillouinpy's constructor
    # already accepts directly, and a subsequent spatial op (indexing, `.mean`)
    # behaves the same as if the arrays had originated in brillouinpy.
    rng = np.random.default_rng(0)
    data = rng.random((2, 2, 10))
    axis = np.linspace(0, 100, 10)
    raman_image = ramanspy.SpectralImage(data, axis)

    image = SpectralImage(raman_image.spectral_data, raman_image.spectral_axis)

    pixel = image[0, 0]
    assert pixel.spectral_data.shape == (10,)
    assert np.allclose(np.ma.filled(pixel.spectral_data), data[0, 0])

    mean_spectrum = image.mean
    assert mean_spectrum.spectral_data.shape == (10,)
    assert np.allclose(np.ma.filled(mean_spectrum.spectral_data), data.reshape(-1, 10).mean(axis=0))
