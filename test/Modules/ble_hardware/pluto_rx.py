import adi
from .base import Receiver
import numpy as np


class PlutoReceiver(Receiver):
    '''
    Receiver using Adalog Devices ADALM_PLUTO SDR
    
    Just records I/Q samples from receiver (does not perform clock and data recovery)
    '''
    def __init__(self, rx_freq: float, symbol_time: float, bt: float, sample_rate: float, ifreq: float, sdr=None, *args, **kwargs):
        self.ifreq = ifreq
        if sdr is None:
            self.sdr = adi.Pluto()
        elif isinstance(sdr, str):
            self.sdr = adi.Pluto(sdr)
        else:
            self.sdr = sdr

        super().__init__(rx_freq, symbol_time, bt, sample_rate, *args, **kwargs)

    def set_rx_freq(self, rx_freq: float) -> None:
        '''
        Set receiver's frequency
        '''
        self.rx_freq = int(rx_freq - self.ifreq)
        self.sdr.rx_lo = self.rx_freq

    def set_rx_gain(self, gain: float=70, mode: str='manual') -> None:
        '''
        Set the receiver's gain and ACG strategy
        '''
        if mode not in ['manual', 'slow_attack', 'fast_attack', 'hybrid']:
            raise ValueError("Gain mode must be one of 'manual', 'slow_attack', 'fast_attack', or 'hybrid'")
        
        if mode == 'manual' and not 0 <= gain <= 70:
            raise ValueError("Gain must be in the range 0 to 70 dB")
        
        self.sdr.gain_control_mode_chan0 = mode
        if mode == 'manual':
            self.sdr.rx_hardwaregain_chan0 = gain
    
    def set_sample_rate(self, sample_rate: int=16_000_000) -> None:
        '''
        Set the sample rate of the Pluto Receiver
        '''
        samples_per_symbol = int(sample_rate * self.symbol_time)
        if abs(samples_per_symbol - (sample_rate * self.symbol_time)) > 0.1:
            raise ValueError("Sample rate is not an integer multiple of symbol rate")
        
        self.sample_rate = sample_rate
        self.sdr.sample_rate = sample_rate
        self.samples_per_symbol = samples_per_symbol

    def receive(self, num_samples: int=20000, clear_buffer: bool=True) -> np.ndarray:
        '''
        Receive samples from the Pluto SDR
        '''
        self.sdr.rx_buffer_size = num_samples
        if clear_buffer:
            self.sdr.rx_destroy_buffer()

        return self.sdr.rx()
    
    def close(self) -> None:
        '''
        Close the receiver
        '''
        self.sdr.rx_destroy_buffer()