# Author: JiangWenbin, 2021.07--2022.05
import torch
import numpy as np
from scipy import fftpack
import librosa
import torch.nn.functional as F
from numpy import pi
from .common import torch_float32, EPS
rtol = 1e-04 # relative tolerance
atol = 1e-05 # absolute tolerance

# frame signal --------------------------------------------------------


def frame_unfold(x, frame_length, hop_length):
    """Frame a signal with given frame size and hop size.
    Args:
        x: [n_batch, n_sample]
    Returns:
        framed_x: [n_batch, frame_length, (n_sample - frame_length + hop_length) // hop_length]
    """
    x = x[:, None, :, None]  # [n_batch, n_sample] --> [n_batch, 1, n_sample, 1]
    return F.unfold(x, kernel_size=(frame_length, 1), stride=(hop_length, 1))


def frame_stride(x, frame_length, hop_length):
    """ Frame a signal with given frame size and hop size.
    Args:
        x: [n_batch, n_sample]
    Returns:
        x_frames: [n_batch, frame_length, 1 + (n_sample - frame_length) // hop_length]
    Ref: 
        https://librosa.org/doc/latest/generated/librosa.util.frame.html
        https://www.tensorflow.org/api_docs/python/tf/signal/frame
        https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe#file-x_tools-py-L1368
    """
    if not torch.is_tensor(x):
        raise TypeError("X must be an tensor, given {}".format(type(x)))
    n_frames = 1 + (x.shape[-1] - frame_length) // hop_length
    stride = list(x.stride())
    shape = list(x.shape)[:-1] + [frame_length, n_frames]
    stride_new = stride + [hop_length * stride[-1]]
    return torch.as_strided(x, size=shape, stride=stride_new)


def frame(x, frame_length, hop_length, method='unfold'):
    """ Slice signal x into frames
    Notice: The end of x will be truncated, this is compatible with librosa.
    Do some padding, if you want to preserve the end of x.
    Args:
        x: [n_batch, n_sample]
    Returns:
        x_frames: [n_batch, frame_length, 1 + (n_sample - frame_length) // hop_length]
    """
    x = torch_float32(x)
    if len(x.shape) < 2:
        raise ValueError("X shape must be [n_bath, n_sample], given {}".format(x.shape))
    if method == 'stride':
        return frame_stride(x, frame_length, hop_length)
    else:
        return frame_unfold(x, frame_length, hop_length)

# aliase
enframe = frame

# unframe signal --------------------------------------------------------

def overlap_add_naive(x_frames, hop_length):
    """ Overlap add frames to reconstruct signal
    Notice: no Hamming window is applied, just overlap and add

    Args:
        x_frames: [n_batch, frame_length, n_frames]
        hop_length: frame shift size

    Returns:
        x: [n_batch, n_sample]

    Ref:
        https://www.tensorflow.org/api_docs/python/tf/signal/overlap_and_add
        https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe#file-x_tools-py-L1458
    """
    if not torch.is_tensor(x_frames):
        raise TypeError("x_frames must be an tensor, given {}".format(type(x_frames)))
    
    n_batch, frame_length, n_frame = x_frames.shape
    n_sample = frame_length + (n_frame - 1) * hop_length
    x = torch.zeros((n_batch, n_sample)).type_as(x_frames)
    start_index = 0

    for i in range(n_frame):
        end_index = start_index + frame_length
        x[:, start_index:end_index] += x_frames[:, :, i]
        start_index += hop_length
    return x


def overlap_add_fold(x_frames, hop_length):
    """This function uses overlap and add to unframe a framed signal.
    Shapes:
        x_frames: [n_batch, frame_length, n_frame]
        returns: [n_batch, frame_length + (n_frame - 1) * hop_length]
    """
    _, frame_length, n_frame = x_frames.shape
    n_sample = frame_length + (n_frame - 1) * hop_length
    x = F.fold(
        x_frames,
        output_size=(n_sample, 1),
        kernel_size=(frame_length, 1),
        stride=(hop_length, 1),
    ).squeeze(-1)
    return x.squeeze(1)


def overlap_add(x_frames, hop_length, method='fold'):
    if method == 'fold':
        return overlap_add_fold(x_frames, hop_length)        
    else:
        return overlap_add_naive(x_frames, hop_length)

# aliase
unframe = overlap_add

# stft and istft -------------------------------------------------------

def get_window(n_fft, win_length=None, window='hann'):
    """get padded window for stft and istft
    """
    if win_length is None:
        win_length = n_fft
    window = getattr(torch, f"{window}_window")(win_length)
    n_pad_left = (n_fft - window.shape[0]) // 2
    n_pad_right = n_fft - window.shape[0] - n_pad_left
    window = F.pad(window, pad=(n_pad_left, n_pad_right))
    window = window.reshape(1, -1, 1)  # reshape to broadcast
    return window


def apply_framing(x, n_fft, hop_length=None, win_length=None, window='hann',
                 center=True, pad_mode='reflect'):
    """apply framing
    x: shape of [n_batch, n_samples]
    return: shape of [n_batch, frame_length, n_frames]
    """
    if center:
        pad = (n_fft // 2, n_fft // 2)
        x = F.pad(x.unsqueeze(1), pad=pad, mode=pad_mode).squeeze(1)

    x_frames = frame(x, frame_length=n_fft, hop_length=hop_length)

    if type(window) is str:
        _window = get_window(n_fft, win_length, window).type_as(x_frames)
        
    return _window * x_frames


def apply_overlap_add(x_frames, n_fft=None, hop_length=None, win_length=None, 
          window='hann', center=True, length=None, window_sum=None):
    """apply window, overlap add, normalize, and fix length
    x_frames: shape of [n_batch, n_fft, n_frames]
    return: shape of [n_batch, n_samples]
    """
    
    if type(window) is str:
        _window = get_window(n_fft, win_length, window).type_as(x_frames)
    
    x = overlap_add(_window * x_frames, hop_length)
    
    if window_sum is None:
        if length:
            padded_length = length + int(n_fft) if center else length
            n_frames = min(x_frames.shape[-1], int(np.ceil(padded_length / hop_length)))
        else:
            n_frames = x_frames.shape[-1]        
        window_sum = librosa.filters.window_sumsquare(
            window=window,
            n_frames=n_frames,
            win_length=win_length,
            n_fft=n_fft,
            hop_length=hop_length
        )
        window_sum = torch_float32(window_sum).type_as(x)               
        
    # Normalize by sum of squared window
    approx_nonzero_indices = window_sum > torch.finfo(torch.float32).tiny
    x[..., approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
    
    # remove the center padding or padding to the length
    if length is None:
        if center:
            x = x[..., int(n_fft // 2) : -int(n_fft // 2)]
    else:
        start = n_fft // 2 if center else 0
        x = fix_length(x[..., start:], size=length)
                    
    return x


def stft(x, n_fft, hop_length=None, win_length=None, window='hann',
         center=True, pad_mode='reflect'):
    """Short-time Fourier transform (STFT). 
    The same as torch.stft and librosa.stft
    """
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    x_frames = apply_framing(x, n_fft, hop_length, win_length, 
                             window, center, pad_mode)
                
    x_stft = torch.fft.rfft(x_frames, dim=-2)

    return x_stft


def istft(x_stft, n_fft=None, hop_length=None, win_length=None, 
          window='hann', center=True, length=None, window_sum=None):
    """Inverse short time Fourier Transform (ISTFT)
    The same as torch.istft and librosa.istft
    """
    if n_fft is None:
        n_fft = 2 * (x_stft.shape[-2] - 1)

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    x_frames = torch.fft.irfft(x_stft, n=n_fft, dim=-2)
    
    x = apply_overlap_add(x_frames, n_fft, hop_length, win_length,
                          window, center, length, window_sum)
        
    return x


# convolution --------------------------------------------------------

def get_fft_size(frame_length, ir_size, power_of_2 = True):
    """Calculate final size for efficient FFT.
    
    Ref: 
        https://github.com/magenta/ddsp/blob/main/ddsp/core.py#L1083

    Args:
        frame_length: Size of the x frame.
        ir_size: Size of the convolving impulse response.
        power_of_2: Constrain to be a power of 2.

    Returns:
        fft_size: Size for efficient FFT.
    """
    convolved_frame_length = ir_size + frame_length - 1
    if power_of_2:
        # Next power of 2.
        fft_size = int(2**np.ceil(np.log2(convolved_frame_length)))
    else:
        fft_size = int(fftpack.helper.next_fast_len(convolved_frame_length))
    return fft_size


def padding(x, frame_length, hop_length):
    """pading the end of x with zeros
    Shapes:
        x: [n_batch, n_sample]    
    Ref:
        https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/ops/signal/shape_ops.py#L166
    """    
    if not torch.is_tensor(x):
        raise TypeError("X must be an tensor, given {}".format(type(x)))
    
    _, n_sample = x.shape    
    n_frames = -(-n_sample // hop_length) # Using double negatives to round up.
    n_padding = frame_length + hop_length * (n_frames - 1) - n_sample
    x_padded = F.pad(x, (0, n_padding), mode='constant', value=0)
    return x_padded


def crop_and_compensate_delay(x, x_size, ir_size, pad_mode, delay_compensation):
    """Crop x output from convolution to compensate for group delay.

    Args:
        x: x after convolution. Shape [n_batch, time_steps].
        x_size: Initial size of the x before convolution.
        ir_size: Size of the convolving impulse response.
        pad_mode: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input x (x_timesteps). For 'valid' the x is extended 
            to include the tail of the impulse response (x_timesteps + ir_timesteps - 1).
        delay_compensation: n to crop from start of output x to compensate
            for group delay of the impulse response. If delay_compensation < 0 it
            defaults to automatically calculating a constant group delay of the
            windowed linear phase filter from frequency_impulse_response().

    Returns:
        Tensor of cropped and shifted x.

    Ref:
        https://github.com/magenta/ddsp/blob/main/ddsp/core.py#L1104
    """
    # Crop the output.
    if pad_mode == 'valid':
        crop_size = ir_size + x_size - 1
    elif pad_mode == 'same':
        crop_size = x_size
    else:
        raise ValueError('pad_mode must be \'valid\' or \'same\', instead '
                        'of {}.'.format(pad_mode))

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = int(x.shape[-1])
    crop = total_size - crop_size
    start = ((ir_size - 1) // 2 - 1 if delay_compensation < 0 else delay_compensation)
    end = crop - start
    return x[:, start:-end]


def fft_convolve(x, impulse_response, pad_mode = 'same', delay_compensation = -1):
    """Filter x with frames of time-varying impulse responses.

    Given x [n_batch, n_sample], and a series of impulse responses [n_batch, 
    ir_frames,  ir_size], splits the x into frames, applies filters, and then 
    overlap-add x  back together. Applies non-windowed non-overlapping STFT/ISTFT 
    to efficiently compute convolution for large impulse response sizes.
    TODO: check if a Hamming window is needed

    Args:
        x: Input speech. Tensor of shape [n_batch, n_sample].
        impulse_response: Finite impulse response to convolve. Can either be a 2-D      
            Tensor of shape [n_batch, ir_size], or a 3-D Tensor of shape [n_batch,
            ir_frames, ir_size]. A 2-D tensor will apply a single linear time-invariant 
            filter to the x. A 3-D Tensor will apply a linear time-varying filter. 
            Automatically chops the x into equally shaped blocks to match ir_frames.
        pad_mode: Either 'valid' or 'same'. For 'same' the final output to be the            
            same size as the input x (n_sample). For 'valid' the x is extended 
            to include the tail of the impulse response (n_sample + ir_size - 1).
        delay_compensation: n to crop from start of output x to compensate
            for group delay of the impulse response. If delay_compensation is less than 
            0 it defaults to automatically calculating a constant group delay of the 
            windowed linear phase filter from frequency_impulse_response().

    Returns:
        x_out: Convolved x. Tensor of shape 
            [n_batch, n_sample + ir_size - 1] ('valid' pad_mode) or shape
            [n_batch, n_sample] ('same' pad_mode).
    
    Ref: 
        https://github.com/magenta/ddsp/blob/main/ddsp/core.py#L1148
        https://github.com/scipy/scipy/blob/v1.7.0/scipy/signal/signaltools.py#L554
    """
    x, impulse_response = torch_float32(x), torch_float32(impulse_response)

    # Get shapes of x.
    n_batch, x_size = x.shape

    # Add a frame dimension to impulse response if it doesn't have one.
    ir_shape = impulse_response.shape
    if len(ir_shape) == 2:
        impulse_response = impulse_response[:, None, :] #expend dimension

    # Broadcast impulse response.
    if ir_shape[0] == 1 and n_batch > 1:
        impulse_response = torch.tile(impulse_response, [n_batch, 1, 1]) #repeat in the fist dimension

    # Get shapes of impulse response.
    n_batch_ir, n_ir_frames, ir_size = impulse_response.shape

    # Validate that batch sizes match.
    if n_batch != n_batch_ir:
        raise ValueError(f'Batch size of x ({n_batch}) != impulse response ({n_batch_ir})')

    # Cut x into frames.
    frame_length = int(np.ceil(x_size / n_ir_frames))
    hop_length = frame_length
    x = padding(x, frame_length, hop_length)
    x_frames = frame(x, frame_length, hop_length) # x_frames: [n_batch, frame_length, n_frames]

    # Check that number of frames match.
    n_x_frames = int(x_frames.shape[-1])
    if n_x_frames != n_ir_frames:
        raise ValueError(f'Frames of x ({n_x_frames}) != impulse response ({n_ir_frames})')

    # Pad and FFT the x and impulse responses.
    fft_size = get_fft_size(frame_length, ir_size)
    # [n_batch, frame_length, n_frames] --> [n_batch, n_frames, frame_length]
    x_frames = x_frames.transpose(1, 2)
    x_fft = torch.fft.rfft(x_frames, n=fft_size)
    ir_fft = torch.fft.rfft(impulse_response, n=fft_size)

    # Multiply the FFTs (same as convolution in time).
    x_ir_fft = torch.multiply(x_fft, ir_fft)

    # Take the IFFT to resynthesize x.
    x_frames_out = torch.fft.irfft(x_ir_fft)
    # [n_batch, n_frames, frame_length] --> [n_batch, frame_length, n_frames]
    x_frames_out = x_frames_out.transpose(1, 2)
    x_out = unframe(x_frames_out, hop_length)

    # Crop and shift the output x.
    return crop_and_compensate_delay(x_out, x_size, ir_size, pad_mode, delay_compensation)


def window_IRs(impulse_response, window_size = 0, causal = False):
    """Apply a window to an impulse response and put in causal form.

    Args:
        impulse_response: A series of impulse responses frames to window, of shape
            [n_batch, n_frames, ir_size].
        window_size: Size of the window to apply in the time domain. If window_size
            is less than 1, it defaults to the impulse_response size.
        causal: Impulse responnse input is in causal form (peak in the middle).

    Returns:
        impulse_response: Windowed impulse response in causal form, with last
            dimension cropped to window_size if window_size is greater than 0 and less
            than ir_size.
    """
    impulse_response = torch_float32(impulse_response)

    # If IR is in causal form, put it in zero-phase form.
    if causal:
        impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

    # Get a window for better time/frequency resolution than rectangular.
    # Window defaults to IR size, cannot be bigger.
    ir_size = int(impulse_response.shape[-1])
    if (window_size <= 0) or (window_size > ir_size):
        window_size = ir_size
    window = torch.hann_window(window_size)

    # Zero pad the window and put in in zero-phase form.
    n_padding = ir_size - window_size
    if n_padding > 0:
        half_i = (window_size + 1) // 2
        window = torch.cat([window[half_i:],
                            torch.zeros([n_padding]),
                            window[:half_i]], axis=0)
    else:
        window = torch.fft.fftshift(window, dim=-1)

    # Apply the window, to get new IR (both in zero-phase form).
    window = torch.broadcast_to(window, impulse_response.shape)
    impulse_response = window * impulse_response

    # Put IR in causal form and trim zero n_padding.
    if n_padding > 0:
        first_half_start = (ir_size - (half_i - 1)) + 1
        second_half_end = half_i + 1
        impulse_response = torch.cat([impulse_response[..., first_half_start:],
                                        impulse_response[..., :second_half_end]],
                                        axis=-1)
    else:
        impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

    return impulse_response


def frequency_impulse_response(magnitudes, window_size = 0):
    """Get windowed impulse responses using the frequency sampling method.

    Follows the approach in:
    https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html
    """
    # Get the IR (zero-phase form).
    magnitudes = torch_float32(magnitudes)
    magnitudes = torch.complex(magnitudes, torch.zeros_like(magnitudes))
    impulse_response = torch.fft.irfft(magnitudes)

    # Window and put in causal form.
    impulse_response = window_IRs(impulse_response, window_size)

    return impulse_response


def frequency_filter(x, magnitudes, window_size = 0, pad_mode = 'same'):
    """Filter x with a finite impulse response filter.

    Args:
        x: Input x. Tensor of shape [n_batch, x_timesteps].
        magnitudes: Frequency transfer curve. Float32 Tensor of shape [n_batch,
            n_frames, n_frequencies] or [n_batch, n_frequencies]. The frequencies 
            of the last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
            f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
            x into equally sized frames to match frames in magnitudes.
        window_size: Size of the window to apply in the time domain. If window_size
            is less than 1, it is set as the default (n_frequencies).
        pad_mode: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input x (x_timesteps). For 'valid' the x is
            extended to include the tail of the impulse response (x_timesteps +
            window_size - 1).

    Returns:
        Filtered x. Tensor of shape
            [n_batch, x_timesteps + window_size - 1] ('valid' pad_mode) or shape
            [n_batch, x_timesteps] ('same' pad_mode).
    """
    impulse_response = frequency_impulse_response(magnitudes, window_size)

    return fft_convolve(x, impulse_response, pad_mode=pad_mode)


# cepstrum --------------------------------------------------------
"""
https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/cepstrum.py
"""

def unwrap(p, discont=pi, axis=-1):
    """
    Args:
        p: shape [n_batch, n_frames, fft_size]
    Ref:        
        https://github.com/numpy/numpy/blob/v1.19.5/numpy/lib/function_base.py#L1488
    """   
    dd = torch.diff(p, axis=axis)
    slice1 = [slice(None, None)] * p.ndim     # full slices
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    ddmod = torch.remainder(dd + pi, 2*pi) - pi
    ddmod[(ddmod == -pi) & (dd > 0)] = pi    
    ph_correct = ddmod - dd
    ph_correct[dd.abs() < discont] = 0    
    up = torch.clone(p)
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up


def phase_unwrap(phase, axis=-1):
    n = phase.shape[axis] # n_fft
    center = 0 if n == 1 else (n + 1) // 2
    unwrapped = unwrap(phase, axis=axis)
    # unwrapped = torch.from_numpy(np.unwrap(phase, axis=axis))
    ndelay = torch.round(unwrapped[..., center] / pi)
    unwrapped -= pi * ndelay[..., None] * torch.arange(n).type_as(phase) / center
    return unwrapped, ndelay


def complex_cepstrum(x, n=None, dim=-1):
    """
    Args:
        x: Real sequence of shape [n_batch, n_frames, frame_length]
        n: Length of Fourier transform, None or int
        dim: The dimension along which to take the one dimensional FFT (default -1)
    Returns:
        ceps: The complex cepstrum of shape [n_batch, n_frames, ceps_size], ceps_size is n
        ndelay: The amount of sample of circular delay added to `x`.
    Ref:
        https://ww2.mathworks.cn/help/signal/ref/cceps.html
    """        
    assert dim == -1, "Only dimension -1 is supported"
    x = torch_float32(x)
    spectrum = torch.fft.fft(x, n=n, dim=dim)
    unwrapped_phase, ndelay = phase_unwrap(torch.angle(spectrum))
    mag = torch.abs(spectrum).clip(EPS) # clip: in case of -inf for logarithm
    log_spectrum = torch.complex(torch.log(mag), unwrapped_phase)
    ceps = torch.real(torch.fft.ifft(log_spectrum, dim=dim))
    return ceps, ndelay


def real_cepstrum(x, n=None, dim=-1):
    """
    Args:
        x: Real sequence of shape [n_batch, n_frames, frame_length]
        n: Length of Fourier transform, None or int
    Return:
        ceps: The real cepstrum of shape [n_batch, n_frames, ceps_size], ceps_size is n
    Ref:
        https://ww2.mathworks.cn/help/signal/ref/rceps.html
    """
    assert dim == -1, "Only dimension -1 is supported"
    x = torch_float32(x)
    spectrum = torch.fft.fft(x, n=n, dim=dim)    
    mag = torch.abs(spectrum).clip(EPS) # clip: in case of -inf for logarithm    
    ceps = torch.fft.ifft(torch.log(mag), dim=dim)
    return ceps.real


def phase_wrap(phase, ndelay=0):
    """Warp the phase
    """
    ndelay = torch_float32(ndelay).type_as(phase)
    n = phase.shape[-1]
    center = (n + 1) // 2
    wrapped = phase + pi * ndelay[..., None] * torch.arange(n).type_as(phase) / center
    return wrapped


def inverse_complex_cepstrum(ceps, ndelay=0):
    """
    Args:
        ceps: complex cepstrum of shape [n_batch, n_frames, ceps_size]
        ndelay: The amount of n of circular delay added to x.
    Return:
        The inverse complex cepstrum of [n_batch, n_frames, ceps_size]
    Ref:
        https://ww2.mathworks.cn/help/signal/ref/icceps.html
    """
    log_spectrum = torch.fft.fft(ceps)
    spectrum = torch.exp(torch.complex(log_spectrum.real, 
                                       phase_wrap(log_spectrum.imag, ndelay)))
    x = torch.fft.ifft(spectrum)
    return x.real


def minimum_phase(x, n=None, dim=-1):
    """
    Args:
        x: Real sequence if shape [n_batch, n_frames, frame_length]
        n: Length of Fourier transform, None or int
    """
    x = torch_float32(x)
    if n is None:
        n = x.shape[-1]
    ceps = real_cepstrum(x, n, dim)
    odd = n % 2
    window = torch.cat((torch.tensor([1.0]), 2.0 * torch.ones((n + odd)// 2 - 1), 
                        torch.ones(1 - odd), torch.zeros((n + odd)// 2 - 1)))
    window = window.type_as(x)
    m = torch.fft.ifft(torch.exp(torch.fft.fft(window * ceps)))

    return m.real

# signal tools --------------------------------------------------------
def circsum(x, n, dim=-1):
    """
    Circular summation of x with length n
    Args:
        x: tensor
        n: window length
    """
    assert dim == -1, "Only dimension -1 is supported"
    n_sample = x.shape[dim]
    assert n < n_sample
    y = torch.zeros_like(x, dtype=torch.float32)
    y[..., 0] = torch.sum(x[..., 0:n], axis=-1)
    # shift from right to left
    for i in range(1, n_sample):
        head = (0 - i) % n_sample
        tail = (0 - i + n) % n_sample
        y[..., i] = y[..., i-1] + x[..., head] - x[..., tail]
    
    return y


def circshift_pad(x, I):
    """
    circular shift x with indices I, shift from left to right
    749 ms ± 74.2 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    """
    [n_batch, n_frames, frame_length] = x.shape
    I = torch.fmod(I, frame_length)    
    x_pad = F.pad(x, (frame_length, frame_length), "circular") # circular padding
    data_list = []
    for i in range(n_batch):
        temp_list = [x_pad[i, j, (frame_length-I[i,j]):(2*frame_length-I[i,j])] for j in range(n_frames)]
        data_list.append(torch.stack(temp_list))
    return torch.stack(data_list)


def circshift_roll(x, I):
    """
    circular shift x with indices I, shift from left to right
    476 ms ± 24.9 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    """
    [n_batch, n_frames, _] = x.shape
    data_list = []
    for i in range(n_batch):
        temp_list = [torch.roll(x[i, j, :], int(I[i, j]), dims=0) for j in range(n_frames)]
        data_list.append(torch.stack(temp_list))
    return torch.stack(data_list)


def circshift(x, I, method='roll'):
    """
    circular shift x with indices I, shift from left to right
    """
    assert x.shape[0:2] == I.shape, "shape error: {} ~= {}".format(x.shape[0:2], I.shape)
    if method == 'roll':
        x_shift = circshift_roll(x, I)
    else:
        x_shift = circshift_pad(x, I)
    assert x.shape == x_shift.shape, "shap error: {} ~= {}".format(x.shape, x_shift.shape)
    return x_shift


def circular_convolve(x, y):
    """
    Circular convolution of framed x and y
    Params:
        x: [n_batch, n_frames, fft_size]
        y: [n_batch, n_frames, fft_size]
    Returns:
        [n_batch, n_frames, fft_size]
    """
    return torch.real(torch.fft.ifft(torch.fft.fft(x)*torch.fft.fft(y)))


def power_spectrum(x, fft_length=256, window_length=256, hop_length=128, 
                   db=True, cutoff=-80):
    """
    x: [n_batch, 1, T]
    returns: [n_batch, FFT // 2 + 1, F]
    """
    window = torch.hann_window(window_length).type_as(x)
    X = torch.stft(x, fft_length, hop_length, window_length, window, return_complex=False)
    X_power = X.pow(2).sum(dim=-1)
    if db is False:
        return X_power    
    X_db = 10 * ((X.pow(2).sum(dim=-1) + EPS).log10())
    X_db[X_db < cutoff] = cutoff
    return X_db