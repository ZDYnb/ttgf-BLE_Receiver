from math import pi, sqrt, log, atan2
import numpy as np
from scipy import signal


def create_gaussian_filter(df: float, samples_per_bit: int, bit_time: float) -> np.ndarray:
    '''
    Create a gaussian filter for GFSK modulation
    '''
    bandwidth_time = 2 * df * bit_time
    alpha = sqrt(log(2) / 2) / bandwidth_time
    t = np.linspace(-bit_time / 2, bit_time / 2, samples_per_bit)
    gaussian = (sqrt(pi) / alpha) * np.exp(-(pi * t / alpha) ** 2)
    return gaussian


def filter_bitstream(bitstream: str, df: float, samples_per_bit: int, bit_time: float) -> np.ndarray:
    '''
    Filter the bitstream using a gaussian filter
    '''
    gaussian = create_gaussian_filter(df, samples_per_bit, bit_time)
    binary_wave = np.repeat([float(int(bit) * 2 - 1) for bit in bitstream], samples_per_bit)
    binary_wave = np.concatenate([np.repeat(float(int(bitstream[0]) * 2 - 1), samples_per_bit // 2), binary_wave, np.repeat(float(int(bitstream[-1]) * 2 - 1), samples_per_bit // 2 - 1)])

    filtered_bitstream = signal.convolve(binary_wave, gaussian, mode="valid")
    filtered_bitstream /= np.max(np.abs(filtered_bitstream))

    return filtered_bitstream


def createFSK(bitstream: str, amp: float, center_freq: float, df: float, samples_per_bit: int, bit_time: float, offset: float=0, initial_phase: float=0, gaussian_filter: bool=True, round: str | None='round'):
    '''
    Create a FSK modulated wave from a bitstream
    '''

    if gaussian_filter:
        # Create gaussian filtered bitstream
        filtered_bitstream = filter_bitstream(bitstream, df, samples_per_bit, bit_time)

        # Create GFSK Wave
        next_t = bit_time / samples_per_bit
        gfsk_wave = np.zeros(len(bitstream) * samples_per_bit, dtype=complex)
        gfsk_wave[0] = np.exp(1j * initial_phase)
        for ix, df_mult in enumerate(filtered_bitstream[:-1]):
            phase_offset = atan2(np.imag(gfsk_wave[ix]), np.real(gfsk_wave[ix]))
            gfsk_wave[ix + 1] = np.exp(1j * (2 * pi * (center_freq + df * df_mult) * next_t + phase_offset))
            
        fsk_wave = gfsk_wave
    else:
        # Create FSK Wave
        fsk_wave = np.zeros(len(bitstream) * samples_per_bit, dtype=complex)
        t = np.linspace(0, bit_time, samples_per_bit)
        next_t = t[-1] + (t[1] - t[0])
        phase_offset = initial_phase
        for ix, val in enumerate(bitstream):
            if val == '1':
                exp_points = np.exp(1j * (2 * pi * (center_freq + df) * t + phase_offset))
                point = np.exp(1j * (2 * pi * (center_freq + df) * next_t + phase_offset))
            else:
                exp_points = np.exp(1j * (2 * pi * (center_freq - df) * t + phase_offset))
                point = np.exp(1j * (2 * pi * (center_freq - df) * next_t + phase_offset))

            fsk_wave[(ix) * samples_per_bit:(ix + 1) * samples_per_bit] = exp_points
            phase_offset = atan2(np.imag(point), np.real(point))
        
    fsk_wave = amp * fsk_wave + (offset + 1j * offset)
    if round:
        if isinstance(round, str) and round.lower() == "truncate":
            fsk_wave = np.array([int(n) for n in fsk_wave.real]) + 1j * np.array([int(n) for n in fsk_wave.imag])
        else:
            fsk_wave = np.round(fsk_wave)
        
    return fsk_wave
