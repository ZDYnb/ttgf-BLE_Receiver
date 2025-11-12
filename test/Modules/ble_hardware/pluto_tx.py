import adi
import os
import sys
from pathlib import Path
from .base import Transmitter

iq_path = os.path.join(Path(__file__).resolve().parent, os.getenv("IQ_PATH", "../phy"))
sys.path.insert(0, iq_path)
from iq import createFSK

class PlutoTransmitter(Transmitter):
    '''
    Transmitter using Analog Devices ADALM_PLUTO SDR

    Inherits from the base Transmitter class
    '''
    def __init__(self, tx_freq: float, symbol_time: float, bt: float, tx_power: float, ifreq: float, sdr=None, *args, **kwargs) -> None:
        self.ifreq = ifreq
        if sdr is None:
            self.sdr = adi.Pluto()
        elif isinstance(sdr, str):
            self.sdr = adi.Pluto(sdr)
        else:
            self.sdr = sdr
        
        super().__init__(tx_freq, symbol_time, bt, tx_power, ifreq, *args, **kwargs)
        self.packet = None
        self.samples = None
        self.sdr.tx_rf_bandwidth = int(4 * max(abs(ifreq - self.df), abs(ifreq + self.df)))

    def set_tx_freq(self, tx_freq: float) -> None:
        '''
        Set the transmit frequency
        '''
        # self.tx_freq = int(tx_freq - self.ifreq)
        self.tx_freq = int(tx_freq)
        self.sdr.tx_lo = self.tx_freq

    def set_packet(self, packet: str) -> None:
        '''
        Set the packet to be transmitted
        '''
        self.sdr.tx_destroy_buffer()
        self.packet = packet
        self.samples = createFSK(self.packet, 2 ** 14, self.ifreq, self.df, samples_per_bit=int(self.sample_rate * self.symbol_time), bit_time=self.symbol_time)
        #self.samples = createFSK(self.packet, 2 ** 14, 0, self.df, samples_per_bit=int(self.sample_rate * self.symbol_time), bit_time=self.symbol_time)

    def set_tx_power(self, power: float) -> None:
        '''
        Set the transmit power
        '''
        if not -90 <= power <= 0:
            raise ValueError("Power must be in the range -90 to 0 dB")
        
        self.sdr.tx_hardwaregain_chan0 = power

    def set_sample_rate(self, sample_rate: int=16_000_000) -> None:
        '''
        Set the sample rate of the Pluto Transmitter
        '''
        samples_per_symbol = int(sample_rate * self.symbol_time)
        if abs(samples_per_symbol - (sample_rate * self.symbol_time)) > 0.1:
            raise ValueError("Sample rate is not an integer multiple of symbol rate")
        
        self.sample_rate = sample_rate
        self.sdr.sample_rate = sample_rate
        self.samples_per_symbol = samples_per_symbol

    def single_transmit(self) -> None:
        '''
        Send a single packet
        '''
        if self.samples is None:
            raise ValueError("No samples to transmit")
        self.sdr.tx(self.samples)

    def repeating_transmit(self) -> None:
        '''
        Send repeating packet
        '''
        if self.samples is None:
            raise ValueError("No samples to transmit")
        self.sdr.tx_cyclic_buffer = True
        self.sdr.tx(self.samples)

    def close(self) -> None:
        '''
        Close the transmitter
        '''
        self.sdr.tx_destroy_buffer()
