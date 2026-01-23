import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, FallingEdge, Timer
import os
import sys
import numpy as np
from math import pi

# Add the Modules directory to Python path
test_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(test_dir, 'Modules'))

# Import BLE helper modules
from phy.iq import createFSK

# Test configuration
SAMPLE_RATE = 16        # Samples per symbol
DATA_WIDTH = 4          # Bits per I/Q sample
fsym = 1_000_000        # 1 MHz symbol rate
clk_freq = 16_000_000   # 16 MHz clock

# Signal parameters
DF = 0.25 * fsym 
BT = 1 / fsym 
ifreq = 1250000  

# Calculate I/Q sample range
max_val = 2 ** (DATA_WIDTH - 1) - 1  # +7
min_val = -2 ** (DATA_WIDTH - 1)      # -8
amp = (max_val - min_val) / 2
offset = (max_val + min_val) / 2


def load_packet(packet_name: str) -> str:
    """Load a packet bitstring from file"""
    packet_path = os.path.join(test_dir, 'Packet_Strings', f'{packet_name}.txt')
    with open(packet_path, 'r') as f:
        return f.read().strip()
    

async def reset_design(dut):
    """Reset the design properly"""
    dut.ui_in.value = 0
    dut.ena.value = 0
    dut.uio_in.value = 37  # Channel 37
    dut.rst_n.value = 0
    
    await ClockCycles(dut.clk, 20)
    dut.rst_n.value = 1
    dut.ena.value = 1
    await ClockCycles(dut.clk, 10)
    
    dut._log.info("Reset complete")


# Main Test Function
@cocotb.test()
async def test_debug_outputs(dut):
    """Test Debug - capture and display all outputs"""
    dut._log.info("=" * 60)
    dut._log.info("TEST: Debug Output Capture")
    dut._log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 62.5, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_design(dut)
    
    # Load packet
    packet = load_packet('example_packet')
    # packet = load_packet('1010101_packet')
    # packet = load_packet('1111111_packet')
    # packet = load_packet('1110101101_packet')
    dut._log.info(f"Packet: {packet[:80]}...")
    
    # Generate signal
    test_signal = createFSK(packet, amp, ifreq, DF, SAMPLE_RATE, BT, offset)
    i_samples = np.real(test_signal)
    q_samples = np.imag(test_signal)
    
    outputs = {
        'sample_idx': [],
        'demod_symbol': [],
        'symbol_clk': [],
        'packet_detected': [],
        'uo_out_raw': [],
        'preamble_detected': [],
        # for dewhitening LFSR state debugging
        'dewhiten_lfsr': [],
        'ena_sync': [],
        'reset_dewhiten_crc': [],
        'rx_buffer_0': [],
        'state_copy': [],
        'nextState_copy': [],
        'acc_addr_matched_copy': []
    }

    
    # Send samples and capture everything
    dut._log.info(f"Sending {len(i_samples)} samples...")

    # Add pipeline warmup samples
    warmup_samples = 200
    i_samples = np.concatenate([np.zeros(warmup_samples), i_samples, np.zeros(warmup_samples)])
    q_samples = np.concatenate([np.zeros(warmup_samples), q_samples, np.zeros(warmup_samples)])

    # at end, add extra samples to allow for packet processing
    extra_samples = 50
    i_samples = np.concatenate([i_samples, np.zeros(extra_samples)])
    q_samples = np.concatenate([q_samples, np.zeros(extra_samples)])

    
    for sample_idx, (i, q) in enumerate(zip(i_samples, q_samples)): 
        # Send sample
        i_val = int(np.clip(i, min_val, max_val))
        q_val = int(np.clip(q, min_val, max_val))
        i_unsigned = i_val & 0xF
        q_unsigned = q_val & 0xF
        dut.ui_in.value = i_unsigned | (q_unsigned << 4)

        # Add channel to uio_in[1:0]
        channel = 0  # 0=Ch37, 1=Ch38, 2=Ch39
        dut.uio_in.value = channel & 0x3  # Mask to 2 bits
        
        await RisingEdge(dut.clk)
        await Timer(60, unit='ns')
        
        # Capture outputs
        uo_val = int(dut.uo_out.value)
        demod_symbol = uo_val & 0x1
        symbol_clk = (uo_val >> 1) & 0x1
        packet_det = (uo_val >> 2) & 0x1
        preamble_det = (uo_val >> 3) & 0x1
        # last 4 bits are dewhitening LFSR state first 4 bits, another 3 bits in uio_out[5:3]
        dewhiten_lfsr = ((uo_val >> 4) & 0x1) #| (((int(dut.uio_out.value) >> 3) & 0x7) << 4)

        ena_sync = (int(dut.uio_out.value) >> 6) & 0x1

        reset_dewhiten_crc = (int(dut.uio_out.value) >> 7) & 0x1

        rx_buffer_0 = (int(dut.uio_out.value) >> 2) & 0x1

        state_copy = (int(dut.uio_out.value) >> 5) & 0x1
        nextState_copy = (int(dut.uio_out.value) >> 4) & 0x1
        acc_addr_matched_copy = (int(dut.uio_out.value) >> 3) & 0x1

        outputs['ena_sync'].append(ena_sync)
    
        outputs['dewhiten_lfsr'].append(dewhiten_lfsr)

        
        # Store
        outputs['sample_idx'].append(sample_idx)
        outputs['demod_symbol'].append(demod_symbol)
        outputs['symbol_clk'].append(symbol_clk)
        outputs['packet_detected'].append(packet_det)
        outputs['uo_out_raw'].append(uo_val)
        outputs['preamble_detected'].append(preamble_det)
        outputs['reset_dewhiten_crc'].append(reset_dewhiten_crc)
        outputs['rx_buffer_0'].append(rx_buffer_0)
        outputs['state_copy'].append(state_copy)
        outputs['nextState_copy'].append(nextState_copy)
        outputs['acc_addr_matched_copy'].append(acc_addr_matched_copy)

        
        # # Log symbol clock edges
        # if symbol_clk and not symbol_clk_prev:
        #     dut._log.info(f"  Symbol CLK @ sample {sample_idx}, bit={demod_symbol}")
        
        # Log packet detection
        if packet_det:
            dut._log.info(f"  PACKET DETECTED @ sample {sample_idx}")

        if preamble_det:
            dut._log.info(f"  PREAMBLE DETECTED @ sample {sample_idx}")
     
        # check state changes
        if sample_idx > 0:
            if state_copy != outputs['state_copy'][-2]:
                dut._log.info(f"  STATE CHANGED to {state_copy} @ sample {sample_idx}")
            if nextState_copy != outputs['nextState_copy'][-2]:
                dut._log.info(f"  NEXT STATE CHANGED to {nextState_copy} @ sample {sample_idx}")
            if acc_addr_matched_copy != outputs['acc_addr_matched_copy'][-2]:
                dut._log.info(f"  ACC ADDR MATCHED CHANGED to {acc_addr_matched_copy} @ sample {sample_idx}")
        
        # check reset_dewhiten_crc changes
        if sample_idx > 0:
            if reset_dewhiten_crc != outputs['reset_dewhiten_crc'][-2]:
                dut._log.info(f"  RESET DEWHITEN CRC CHANGED to {reset_dewhiten_crc} @ sample {sample_idx}")

       
    # Save to file and plot
    import csv
    with open('output_capture.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'demod_symbol', 'symbol_clk', 'packet_detected', 'uo_out_raw'])
        for i in range(len(outputs['sample_idx'])):
            writer.writerow([
                outputs['sample_idx'][i],
                outputs['demod_symbol'][i],
                outputs['symbol_clk'][i],
                outputs['packet_detected'][i],
                outputs['uo_out_raw'][i],
                outputs['preamble_detected'][i],
                outputs['dewhiten_lfsr'][i],
                outputs['ena_sync'][i],
                outputs['reset_dewhiten_crc'][i],
                outputs['acc_addr_matched_copy'][i]
            ])
    dut._log.info("Saved outputs to output_capture.csv")

    # Create plots (non-interactive, save to file)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as _np

        samples = _np.array(outputs['sample_idx'])
        demod = _np.array(outputs['demod_symbol'])
        symclk = _np.array(outputs['symbol_clk'])
        pktdet = _np.array(outputs['packet_detected'])
        raw = _np.array(outputs['uo_out_raw'])
        preamble = _np.array(outputs['preamble_detected'])
        
        dewhiten_lfsr = _np.array(outputs['dewhiten_lfsr'])
        ena_sync = _np.array(outputs['ena_sync'])

        reset_dewhiten_crc = _np.array(outputs['reset_dewhiten_crc'])
        

        fig, axes = plt.subplots(10, 1, figsize=(10, 20), sharex=True)
        axes[0].step(samples, demod, where='post', color='C0')
        axes[0].set_ylabel('demod_symbol')
        axes[0].grid(True)

        axes[1].step(samples, symclk, where='post', color='C1')
        axes[1].set_ylabel('symbol_clk')
        axes[1].grid(True)

        axes[2].step(samples, pktdet, where='post', color='C2')
        axes[2].set_ylabel('packet_detected')
        axes[2].grid(True)

        axes[3].plot(samples, raw, '.', markersize=2, color='C3')
        axes[3].set_ylabel('uo_out_raw')
        axes[3].set_xlabel('sample')
        axes[3].grid(True)

        axes[4].step(samples, preamble, where='post', color='C4')
        axes[4].set_ylabel('preamble_detected')
        axes[4].set_xlabel('sample')
        axes[4].grid(True)

        axes[5].step(samples, dewhiten_lfsr, where='post', color='C5')
        axes[5].set_ylabel('dewhiten_lfsr')
        axes[5].set_xlabel('sample')
        axes[5].grid(True)

        axes[6].step(samples, ena_sync, where='post', color='C6')
        axes[6].set_ylabel('ena_sync')
        axes[6].set_xlabel('sample')
        axes[6].grid(True)

        axes[7].step(samples, reset_dewhiten_crc, where='post', color='C7')
        axes[7].set_ylabel('reset_dewhiten_crc')
        axes[7].set_xlabel('sample')
        axes[7].grid(True)

        axes[8].step(samples, outputs['rx_buffer_0'], where='post', color='C8')
        axes[8].set_ylabel('rx_buffer_0')
        axes[8].set_xlabel('sample')
        axes[8].grid(True)

        axes[9].step(samples, outputs['acc_addr_matched_copy'], where='post', color='C9')
        axes[9].set_ylabel('acc_addr_matched_copy')
        axes[9].set_xlabel('sample')
        axes[9].grid(True)

        plt.tight_layout()
        plt.savefig('output_capture.png', dpi=150)
        plt.close(fig)
        dut._log.info("Saved plot to output_capture.png")
    except Exception as e:
        dut._log.warning(f"Could not create plot: {e}")
    # Print summary
    symbol_edges = [i for i, (curr, prev) in enumerate(zip(outputs['symbol_clk'][1:], outputs['symbol_clk'][:-1])) 
                    if curr and not prev]
    recovered_bits = [outputs['demod_symbol'][i] for i in symbol_edges]
    received_buffer_0 = [outputs['rx_buffer_0'][i] for i in symbol_edges]   
    
    dut._log.info(f"\nSUMMARY:")
    dut._log.info(f"  Total samples: {len(outputs['sample_idx'])}")
    dut._log.info(f"  Symbol clock edges: {len(symbol_edges)}")
    dut._log.info(f"  Recovered bits: {''.join(map(str, recovered_bits))}")
    dut._log.info(f"  Original packet: {packet}")
    dut._log.info(f"  Packet detections: {sum(outputs['packet_detected'])}")
    dut._log.info(f"  Preamble detections: {sum(outputs['preamble_detected'])}")
    # summary buffer_0
    dut._log.info(f"  Buffer_0 {''.join(map(str, received_buffer_0))}")

    if len(symbol_edges) > 1:
        periods = np.diff(symbol_edges)
        dut._log.info(f"  Symbol periods: {periods}")
        dut._log.info(f"  Average period: {np.mean(periods):.2f} (expected {SAMPLE_RATE})")

    dut._log.info(" TEST COMPLETE - Check output_capture.csv for details")
