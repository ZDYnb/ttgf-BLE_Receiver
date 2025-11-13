
from time import sleep
from threading import current_thread
from typing import Optional
class Transmitter:
    def __init__(self, tx_freq: float, symbol_time: float, bt: float, tx_power: float, *args, df: Optional[float]=None, **kwargs):
        self.set_tx_freq(tx_freq)
        self.set_tx_power(tx_power)
        self.symbol_time = symbol_time
        self.bt = bt
        if df is not None:
            self.df = df
        else:
            self.df = bt / (symbol_time * 2)
        self.sample_rate = 16_000_000 # Default sample rate
        self.packet = ""

    def set_tx_power(self, tx_power: float):
        raise NotImplementedError("set_tx_power must be implemented by subclass")

    def set_tx_freq(self, tx_freq: float):
        raise NotImplementedError("set_tx_freq must be implemented by subclass")

    def set_packet(self, packet: str):
        raise NotImplementedError("set_packet must be implemented by subclass")

    def transmit(self, cycle_time=1e-3, cycles= 1):
        if len(self.packet) == 0:
            raise ValueError("No packet to transmit")

        if cycle_time > 0:
            sleep_time = cycle_time - (self.symbol_time * len(self.packet))
        else:
            sleep_time = 0

        if cycles is None:
            t = current_thread()
            t.alive = True

            while t.alive:
                self.single_transmit()
                sleep(sleep_time)
        else:
            for _ in range(cycles):
                self.single_transmit()
                sleep(sleep_time)


    def single_transmit(self):
        raise NotImplementedError("single_transmit must be implemented by subclass")

    def close(self):
        raise NotImplementedError("close must be implemented by subclass")

class Receiver:
    def __init__(self, rx_freq: float, symbol_time: float, bt: float, samples_per_bit: float, *args, df: Optional[float]=None, **kwargs):
        self.set_rx_freq(rx_freq)
        self.symbol_time = symbol_time
        self.bt = bt
        if df is not None:
            self.df = df
        else:
            self.df = bt / (symbol_time * 2)
        self.samples_per_bit = samples_per_bit

    def set_rx_freq(self, rx_freq: float):
        raise NotImplementedError("set_rx_freq must be implemented by subclass")

    def receive(self, num_samples: int):
        raise NotImplementedError("receive must be implemented by subclass")

    def close(self):
        raise NotImplementedError("close must be implemented by subclass")


