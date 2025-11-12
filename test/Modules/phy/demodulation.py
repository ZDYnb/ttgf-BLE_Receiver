from math import pi, atan2, sin, cos, ceil, log2, isfinite
from functools import reduce
import numpy as np
from scipy import signal
import numpy as np
from collections import deque, defaultdict
from operator import xor
from typing import Generator, List 
from .iq import createFSK


def genTemplates(center_freq_templates: float, df_templates: float, samples_per_bit: int, bit_time: float, fsk_amp: int, fsk_offset: float) -> np.typing.NDArray[np.float64]:
    '''
    Generate GFSK Templates
    '''
    t0 = createFSK('0', fsk_amp, center_freq_templates, df_templates, samples_per_bit, bit_time, fsk_offset)
    t1 = createFSK('1', fsk_amp, center_freq_templates, df_templates, samples_per_bit, bit_time, fsk_offset)
    
    return t0, t1


def correlate(received_sim: np.typing.NDArray[np.complex128], template: np.typing.NDArray[np.complex128]) -> np.typing.NDArray[np.float64]:
    '''
    Correlate a signal with a template
    '''
    i_i_corr = 0
    q_i_corr = 0
    i_q_corr = 0
    q_q_corr = 0
    for r, t in zip(received_sim, template):
        i_i_corr += np.real(r) * np.real(t)
        q_i_corr += np.imag(r) * np.real(t)
        i_q_corr += np.real(r) * np.imag(t)
        q_q_corr += np.imag(r) * np.imag(t)

    return [(i_i_corr ** 2) + (q_i_corr ** 2) + (i_q_corr ** 2) + (q_q_corr ** 2)]


def matched_filter_score_gen(adc_samples: np.typing.NDArray[np.complex128], center_freq: float, df: float, samples_per_bit: int, bit_time: float, fsk_amp: int, output_type: type=bool) -> Generator[bool | tuple[float, float], None, None]:
    '''
    Generator for matched filter output bit (or scores)
    '''
    low_template, high_template = genTemplates(center_freq, df, samples_per_bit, bit_time, 2 * fsk_amp, 0)
    rx_buffer = deque(np.array([0] * samples_per_bit, dtype=complex), maxlen=samples_per_bit)

    for i in range(len(adc_samples) + samples_per_bit):
        rx_buffer.append(adc_samples[i] if i < len(adc_samples) else 0)
            
        low_score, high_score = correlate(rx_buffer, low_template)[0], correlate(rx_buffer, high_template)[0]
        yield low_score < high_score if output_type == bool else (low_score, high_score)


def matched_filter_score_gen_yield(center_freq: float, df: float, samples_per_bit: int, bit_time: float, fsk_amp: int, output_type: type=bool) -> Generator[bool | tuple[float, float], np.complex128, None]:
    '''
    Generator for matched filter output bit (or scores) that uses yield to pass in samples
    '''
    low_template, high_template = genTemplates(center_freq, df, samples_per_bit, bit_time, 2 * fsk_amp, 0)
    rx_buffer = deque(np.array([0] * samples_per_bit, dtype=complex), maxlen=samples_per_bit)

    while True:
        low_score, high_score = correlate(rx_buffer, low_template)[0], correlate(rx_buffer, high_template)[0]
        sample = yield low_score < high_score if output_type == bool else (low_score, high_score)
        rx_buffer.append(sample)


def preamble_detection_gen(samples_per_bit: int, transition_error: int=1, preamble_len: int=8) -> Generator[bool, bool, None]:
    '''
    Generator to detect a BLE Preamble from matched filter output
    '''
    buffer_len = (preamble_len - 1) * samples_per_bit + 2 * transition_error
    transition_buffer = deque([False] * buffer_len, maxlen=buffer_len)
    last_bit = False

    while True:
        mf_output = yield transition_buffer[0] and all(reduce(xor, [transition_buffer[ix] for ix in range(n-transition_error, n+transition_error+1)]) for n in range(samples_per_bit, buffer_len, samples_per_bit))
        transition_buffer.appendleft(mf_output ^ last_bit)
        last_bit = mf_output


def clock_recovery_if_correction(scum_if: float, samples_per_symbol: int) -> tuple[float, float]:
    '''
    Compute the correction for the S-Curve due to non-zero IF
    '''
    # Calculate the alphabets for a 0 and non-zero IF
    alphabet = np.array([-1, 1], dtype=float)
    low_cycles, high_cycles = scum_if - 0.25, scum_if + 0.25
    low_cycles *= 2
    high_cycles *= 2
    scum_alphabet = alphabet + low_cycles + high_cycles

    a_i = np.array([0, 1], dtype=int)
    a_ideal = alphabet[a_i]
    a_scum = scum_alphabet[a_i]
    a_ideal = np.concatenate((alphabet[:], alphabet[::-1]))
    a_scum = np.concatenate((scum_alphabet[:], scum_alphabet[::-1]))

    # Compute the phase modulation
    q, h = 0.5, 0.5
    phi_t_ideal = np.array([0], dtype=float)
    phi_t_ideal = np.concatenate([phi_t_ideal, 2 * pi * h * np.cumsum([a * q for a in a_ideal])])

    phi_t_scum = np.array([0], dtype=float)
    phi_t_scum = np.concatenate([phi_t_scum, 2 * pi * h * np.cumsum([a * q for a in a_scum])])

    # Compute z(t)
    z_t_ideal = np.array([])
    z_t_scum = np.array([])
    for i in range(1, len(phi_t_ideal)):
        z_t_ideal = np.concatenate([z_t_ideal, np.exp(1j * (np.linspace(phi_t_ideal[i - 1], phi_t_ideal[i], samples_per_symbol)))])
        z_t_scum = np.concatenate([z_t_scum, np.exp(1j * (np.linspace(phi_t_scum[i - 1], phi_t_scum[i], samples_per_symbol)))])

    # Compute c(t)
    c_t_ideal = (z_t_ideal[samples_per_symbol:] ** 2) * np.conj(z_t_ideal[:-samples_per_symbol] ** 2)
    c_t_scum = (z_t_scum[samples_per_symbol:] ** 2) * np.conj(z_t_scum[:-samples_per_symbol] ** 2)

    # Compute s-curves
    deriv_ideal = -np.diff(c_t_ideal)
    deriv_scum = -np.diff(c_t_scum)

    # Compute correction factor
    phase_offset = atan2(np.imag(deriv_scum[0]), np.real(deriv_scum[0])) - atan2(np.imag(deriv_ideal[0]), np.real(deriv_ideal[0]))

    # Check to make sure correction factor works
    corrected_deriv_scum = np.exp(-1j * phase_offset) * deriv_scum
    assert np.max(np.abs(deriv_ideal - corrected_deriv_scum)) < 0.001, f"Calculated correction factor {phase_offset} is incorrect"

    # Compute the coefficients of the linear combination of Real and Imaginary components rotated by the correction factor
    re_mult = cos(phase_offset)
    im_mult = -sin(phase_offset)
    corrected_deriv_scum = (re_mult + 1j * im_mult) * deriv_scum
    assert np.max(np.abs(deriv_ideal - corrected_deriv_scum)) < 0.001, f"Calculated linear combination ({re_mult}, {im_mult}) is incorrect"

    # Return the real and imaginary correction coefficients
    return (-re_mult, -im_mult)


def calc_gamma(adc_depth: float, samples_per_symbol: int) -> float:
    if isfinite(adc_depth):
        return (4 * 5e-3 * (samples_per_symbol ** 2)) / (2 ** (4 * adc_depth + 2))
    else:
        return (4 * 5e-3 * (samples_per_symbol ** 2)) / (2 ** 2)


def clock_recovery_gen(ifreq: float, symbol_time: float, samples_per_symbol: int, adc_depth: float, high_pos: int) -> Generator[tuple[bool, dict], tuple[int, int, bool], None]:
    '''
    Generator to recover the clock of a (G)FSK signal
    '''
    I_k = deque(np.zeros(samples_per_symbol + 3, dtype=int), maxlen=samples_per_symbol + 3)
    Q_k = deque(np.zeros(samples_per_symbol + 3, dtype=int), maxlen=samples_per_symbol + 3)
    shift_counter = 0
    gamma = calc_gamma(adc_depth, samples_per_symbol)
    dtau, i_1, q_1, i_2, q_2, i_3, q_3, i_4, q_4 = [0] * 9

    # sample_pos = high_pos
    sample_pos = (samples_per_symbol - 1) // 2 - 2
    error_calc_counter = 0
    # update_data = shift_counter == sample_pos
    update_data = ((shift_counter % samples_per_symbol) == ((samples_per_symbol - dtau - 3) % samples_per_symbol)) or (error_calc_counter == 3)


    re_mult, im_mult = clock_recovery_if_correction(ifreq * symbol_time, samples_per_symbol)

    while True:
        i_data, q_data, preamble_detected = yield update_data, locals()

        I_k.append(i_data)
        Q_k.append(q_data)        

        i_1 = I_k[0 + samples_per_symbol]
        q_1 = Q_k[0 + samples_per_symbol]
        i_2 = I_k[0]
        q_2 = Q_k[0]
        i_3 = I_k[2 + samples_per_symbol]
        q_3 = Q_k[2 + samples_per_symbol]
        i_4 = I_k[2]
        q_4 = Q_k[2]

        re1 = (i_1*i_1 - q_1*q_1)  * (i_2*i_2 - q_2*q_2) + 4*(i_1*q_1*i_2*q_2)
        re2 = (i_3*i_3 - q_3*q_3)  * (i_4*i_4 - q_4*q_4) + 4*(i_3*q_3*i_4*q_4)
        im1 = 2 * ((i_2 * i_2 * i_1 * q_1) + (q_1 * q_1 * i_2 * q_2) - (i_1 * i_1 * i_2 * q_2) - (q_2 * q_2 * i_1 * q_1))
        im2 = 2 * ((i_4 * i_4 * i_3 * q_3) + (q_3 * q_3 * i_4 * q_4) - (i_3 * i_3 * i_4 * q_4) - (q_4 * q_4 * i_3 * q_3))

        y1 = (re_mult * re1) + (im_mult * im1)
        y2 = (re_mult * re2) + (im_mult * im2)

        e_k = y2 - y1

        do_error_calc = ((shift_counter % samples_per_symbol) == ((samples_per_symbol - dtau - 1) % samples_per_symbol)) or (error_calc_counter == 1)

        if do_error_calc:
            shift_counter = 0
            dtau = round(e_k * gamma * (samples_per_symbol // 2))
        else:
            shift_counter += 1

        if error_calc_counter != 0:
            error_calc_counter = error_calc_counter - 1
        elif preamble_detected:
            # error_calc_counter = (samples_per_symbol >> 1) - sample_pos
            error_calc_counter = (samples_per_symbol - 1) // 2 + 3

        # update_data = (shift_counter) % samples_per_symbol == (sample_pos) % samples_per_symbol
        update_data = ((shift_counter % samples_per_symbol) == ((samples_per_symbol - dtau - 3) % samples_per_symbol)) or (error_calc_counter == 3)


def clock_recovery_gen_old(ifreq: float, symbol_time: float, samples_per_bit: int, e_k_shift: int, tau_shift: int, high_pos: int) -> Generator[tuple[bool, dict], tuple[int, int, bool], None]:
    '''
    Generator to recover the clock of a (G)FSK signal
    '''
    I_k = deque(np.zeros(samples_per_bit + 3, dtype=int), maxlen=samples_per_bit + 3)
    Q_k = deque(np.zeros(samples_per_bit + 3, dtype=int), maxlen=samples_per_bit + 3)
    shift_counter = -1
    tau_int, tau_int_1, tau, tau_1, dtau, i_1, q_1, i_2, q_2, i_3, q_3, i_4, q_4 = [0] * 13

    sample_pos = high_pos
    error_calc_counter = 0
    update_data = shift_counter == sample_pos
    do_error_calc = ((shift_counter % samples_per_bit) == ((samples_per_bit + dtau - 1) % samples_per_bit)) or (error_calc_counter == 1)

    re_mult, im_mult = clock_recovery_if_correction(ifreq * symbol_time, samples_per_bit)

    while True:
        i_data, q_data, preamble_detected = yield update_data, locals()

        if do_error_calc:
            shift_counter = 0
            dtau = tau_1 - tau
            tau_int_1 = tau_int
            tau_1 = tau
        else:
            shift_counter += 1

        if error_calc_counter != 0:
            error_calc_counter = error_calc_counter - 1
        elif preamble_detected:
            error_calc_counter = (samples_per_bit >> 1) - sample_pos

        I_k.append(i_data)
        Q_k.append(q_data)        

        i_1 = I_k[0 + samples_per_bit]
        q_1 = Q_k[0 + samples_per_bit]
        i_2 = I_k[0]
        q_2 = Q_k[0]
        i_3 = I_k[2 + samples_per_bit]
        q_3 = Q_k[2 + samples_per_bit]
        i_4 = I_k[2]
        q_4 = Q_k[2]

        re1 = (i_1*i_1 - q_1*q_1)  * (i_2*i_2 - q_2*q_2) + 4*(i_1*q_1*i_2*q_2)
        re2 = (i_3*i_3 - q_3*q_3)  * (i_4*i_4 - q_4*q_4) + 4*(i_3*q_3*i_4*q_4)
        im1 = 2 * ((i_2 * i_2 * i_1 * q_1) + (q_1 * q_1 * i_2 * q_2) - (i_1 * i_1 * i_2 * q_2) - (q_2 * q_2 * i_1 * q_1))
        im2 = 2 * ((i_4 * i_4 * i_3 * q_3) + (q_3 * q_3 * i_4 * q_4) - (i_3 * i_3 * i_4 * q_4) - (q_4 * q_4 * i_3 * q_3))

        y1 = (re_mult * re1) + (im_mult * im1)
        y2 = (re_mult * re2) + (im_mult * im2)

        e_k = round(y1 - y2)
        e_k_shifted = e_k >> e_k_shift
        tau_int = tau_int_1 - e_k_shifted
        tau = tau_int >> tau_shift

        do_error_calc = ((shift_counter % samples_per_bit) == ((samples_per_bit + dtau - 1) % samples_per_bit)) or (error_calc_counter == 1)
        update_data = shift_counter == sample_pos


def clock_recovery_gen_mf(samples_per_symbol: int) -> Generator[tuple[bool, dict], bool, None]:
    sample_counter = 0
    p_mf_bit = False
    while True:
        mf_bit = yield sample_counter == (samples_per_symbol - 1) // 2, locals()
        # mf_bit = yield sample_counter == 0, locals()

        if mf_bit ^ p_mf_bit:
            sample_counter = -1

        p_mf_bit = mf_bit
        sample_counter += 1
        sample_counter %= samples_per_symbol


def cdr(ifreq: float, df: float, symbol_rate: float, samples_per_bit: int, adc_data: np.typing.NDArray[np.complex128], fsk_amp: int, search_for: np.typing.NDArray[int], e_k_shift: int, tau_shift: int, high_pos: int, mf_clock_rec: bool=False) -> List[List[int]]:
    '''
    Perform clock and data recovery on a signal
    '''
    last_clock = 1
    clock_period = None
    period_counts = defaultdict(int)

    if mf_clock_rec:
        clocks_generated = clock_recovery_gen_mf(samples_per_bit)
        clocks_generated.send(None)
        clock_val = 0
    else:
        preamble_detect = preamble_detection_gen(samples_per_bit)
        preamble_detect.send(None)
        clocks_generated = clock_recovery_gen_old(ifreq, symbol_rate, samples_per_bit, e_k_shift, tau_shift, high_pos)
        # clocks_generated = clock_recovery_gen(ifreq, symbol_rate, samples_per_bit, 4, high_pos)
        clocks_generated.send(None)

    datastream = np.array([], dtype=int)
    high_clocks = np.array([], dtype=int)
    preamble = False
    for ix, (sample, mf_bit) in enumerate(zip(adc_data, matched_filter_score_gen(adc_data, ifreq, df, samples_per_bit, symbol_rate, fsk_amp))):
        if not mf_clock_rec:
            clock_val, _ = clocks_generated.send((int(sample.real), int(sample.imag), preamble))
            preamble = preamble_detect.send(mf_bit)

        if clock_val == 1 and last_clock == 0:
            high_clocks = np.concatenate((high_clocks, [ix]))
            bin_val = int(mf_bit)
                
            datastream = np.concatenate((datastream, [bin_val]))

            if clock_period is not None:
                period_counts[clock_period] += 1

            clock_period = 0

        last_clock = clock_val
        if clock_period is not None:
            clock_period += 1
        
        if mf_clock_rec:
            clock_val, _ = clocks_generated.send(mf_bit)

    if mf_clock_rec and clock_val == 1 and last_clock == 0:
        high_clocks = np.concatenate((high_clocks, [ix]))
        datastream = np.concatenate((datastream, [int(mf_bit)]))

        if clock_period is not None:
            period_counts[clock_period] += 1

    if len(datastream) < len(search_for):
        return [], high_clocks, len(search_for), datastream

    vals = np.array([])
    for delta, count in period_counts.items():
        vals = np.concatenate((vals, [delta] * count))

    return find_matches(datastream, search_for), high_clocks, np.sqrt(np.mean((samples_per_bit - vals) ** 2)), datastream


def find_matches(datastream: np.typing.NDArray[int], search_for: np.typing.NDArray[int]) -> List[List[int]]:
    '''
    Find potential occurrances of search_for in datastream
    '''
    search_correlation = signal.correlate(datastream, search_for) + signal.correlate(datastream - 1, search_for - 1)
    match_ixs = np.fromiter((ix for ix, val in enumerate(search_correlation) if (val * 4 / 3) > len(search_for) and ix >= len(search_for) - 1), dtype=int)
    match_vals = search_correlation[match_ixs]
    match_ixs -= len(search_for) - 1

    return list(zip(match_vals, match_ixs))
