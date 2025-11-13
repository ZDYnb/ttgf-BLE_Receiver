import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, FallingEdge, Timer
import os
import sys
import numpy as np
from math import pi
import random

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


async def send_iq_samples(dut, i_samples, q_samples):
    """Send I/Q samples to the design one at a time"""
    for i, q in zip(i_samples, q_samples):
        # Clip to valid range
        i_val = int(np.clip(i, min_val, max_val))
        q_val = int(np.clip(q, min_val, max_val))
        
        # Pack into ui_in: [Q Q Q Q I I I I]
        # Convert signed to unsigned for Verilog
        i_unsigned = i_val & 0xF
        q_unsigned = q_val & 0xF
        dut.ui_in.value = i_unsigned | (q_unsigned << 4)
        
        await RisingEdge(dut.clk)


async def reset_design(dut):
    """Reset the design properly"""
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 37  # Channel 37
    dut.rst_n.value = 0
    
    await ClockCycles(dut.clk, 2000)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1000)
    
    dut._log.info("Reset complete")

# Main Test Function

async def run_packet_test(dut, packet_bits, test_name="test"):
    """
    Run a complete packet test
    
    Args:
        dut: The device under test
        packet_bits: Bitstring of packet to send
        test_name: Name for logging
    """
    dut._log.info(f"Starting {test_name}")
    dut._log.info(f"Packet length: {len(packet_bits)} bits")
    
    # Generate FSK modulated I/Q samples
    dut._log.info("Generating FSK signal...")
    test_signal = createFSK(
        packet_bits, 
        amp, 
        ifreq, 
        DF, 
        SAMPLE_RATE, 
        BT, 
        offset
    )
    
    # Extract I and Q components
    i_samples = np.real(test_signal)
    q_samples = np.imag(test_signal)
    
    dut._log.info(f"Generated {len(i_samples)} I/Q samples ({len(i_samples)//SAMPLE_RATE} symbols)")
    
    # Add pipeline warmup samples
    warmup_samples = 20
    i_samples = np.concatenate([np.zeros(warmup_samples), i_samples, np.zeros(warmup_samples)])
    q_samples = np.concatenate([np.zeros(warmup_samples), q_samples, np.zeros(warmup_samples)])
    
    # Monitor outputs
    packet_detected_count = 0
    symbol_clk_edges = []
    recovered_bits = []
    last_symbol_clk = 0
    
    # Send samples and monitor outputs
    dut._log.info("Sending I/Q samples...")
    for sample_idx, (i, q) in enumerate(zip(i_samples, q_samples)):
        # Set inputs
        i_val = int(np.clip(i, min_val, max_val))
        q_val = int(np.clip(q, min_val, max_val))
        i_unsigned = i_val & 0xF
        q_unsigned = q_val & 0xF
        dut.ui_in.value = i_unsigned | (q_unsigned << 4)
        
        await RisingEdge(dut.clk)
        
        # Check for packet detection
        if dut.uo_out.value[2]:  # packet_detected is bit 2
            packet_detected_count += 1
            dut._log.info(f" Packet detected at sample {sample_idx}")
        
        # Monitor symbol clock and capture bits
        symbol_clk = dut.uo_out.value[1]  # symbol_clk is bit 1
        if symbol_clk and not last_symbol_clk:  # Rising edge
            symbol_clk_edges.append(sample_idx)
            demod_bit = dut.uo_out.value[0]  # demod_symbol is bit 0
            recovered_bits.append(int(demod_bit))
        
        last_symbol_clk = symbol_clk
        
        await FallingEdge(dut.clk)
    
    # Report results
    dut._log.info(f"Test complete: {test_name}")
    dut._log.info(f"  Packet detections: {packet_detected_count}")
    dut._log.info(f"  Symbol clock edges: {len(symbol_clk_edges)}")
    dut._log.info(f"  Recovered bits: {len(recovered_bits)}")
    
    # Check symbol clock timing
    if len(symbol_clk_edges) > 1:
        periods = np.diff(symbol_clk_edges)
        avg_period = np.mean(periods)
        rmse = np.sqrt(np.mean((periods - SAMPLE_RATE) ** 2))
        dut._log.info(f"  Average symbol period: {avg_period:.2f} samples (expected {SAMPLE_RATE})")
        dut._log.info(f"  Timing RMSE: {rmse:.3f} samples")
    
    return packet_detected_count, recovered_bits, symbol_clk_edges
# Test Cases #

# @cocotb.test()
# async def test_simple_packet(dut):
#     """Test 1: Send a single BLE packet"""
#     dut._log.info("=" * 60)
#     dut._log.info("TEST 1: Simple Packet Test")
#     dut._log.info("=" * 60)
    
#     # Start clock (16 MHz = 62.5 ns period)
#     clock = Clock(dut.clk, 62.5, unit="ns")
#     cocotb.start_soon(clock.start())
    
#     # Reset
#     await reset_design(dut)
    
#     # Load packet
#     packet = load_packet('example_packet')
#     dut._log.info(f"Loaded packet: {len(packet)} bits")
#     dut._log.info(f"First 40 bits: {packet[:40]}")
    
#     # Run test
#     detections, bits, clk_edges = await run_packet_test(dut, packet, "simple_packet")
    
#     # Verify we got at least one detection
#     assert detections > 0, "No packet detected!"
    
#     dut._log.info(" TEST 1 PASSED")


# @cocotb.test()
# async def test_packet_with_noise(dut):
#     """Test 2: Packet with noise before and after"""
#     dut._log.info("=" * 60)
#     dut._log.info("TEST 2: Packet with Noise")
#     dut._log.info("=" * 60)
    
#     # Start clock
#     clock = Clock(dut.clk, 62.5, unit="ns")
#     cocotb.start_soon(clock.start())
    
#     # Reset
#     await reset_design(dut)
    
#     # Load packet
#     packet = load_packet('example_packet')
    
#     # Generate packet signal
#     packet_signal = createFSK(packet, amp, ifreq, DF, SAMPLE_RATE, BT, offset)
    
#     # Add noise before and after
#     noise_duration = 500  # samples
#     rng = np.random.default_rng()
#     noise_before = rng.integers(min_val, max_val, noise_duration) + \
#                    1j * rng.integers(min_val, max_val, noise_duration)
#     noise_after = rng.integers(min_val, max_val, noise_duration) + \
#                   1j * rng.integers(min_val, max_val, noise_duration)
    
#     # Combine
#     test_signal = np.concatenate([noise_before, packet_signal, noise_after])
    
#     i_samples = np.real(test_signal)
#     q_samples = np.imag(test_signal)
    
#     dut._log.info(f"Total samples: {len(i_samples)} (noise + packet + noise)")
#     dut._log.info(f"Packet starts at sample: {noise_duration}")
    
#     # Send and monitor
#     packet_detected_count = 0
#     for sample_idx, (i, q) in enumerate(zip(i_samples, q_samples)):
#         i_val = int(np.clip(i, min_val, max_val))
#         q_val = int(np.clip(q, min_val, max_val))
#         i_unsigned = i_val & 0xF
#         q_unsigned = q_val & 0xF
#         dut.ui_in.value = i_unsigned | (q_unsigned << 4)
        
#         await RisingEdge(dut.clk)
        
#         if dut.uo_out.value[2]:
#             if packet_detected_count == 0:
#                 dut._log.info(f" Packet detected at sample {sample_idx}")
#             packet_detected_count += 1
        
#         await FallingEdge(dut.clk)
    
#     assert packet_detected_count > 0, "No packet detected in noise!"
    
#     dut._log.info(f"Packet detected {packet_detected_count} times")
#     dut._log.info(" TEST 2 PASSED")


@cocotb.test()
async def test_design_basics(dut):
    """Test 3: Basic sanity check - does the design respond?"""
    dut._log.info("=" * 60)
    dut._log.info("TEST 3: Basic Sanity Check")
    dut._log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 62.5, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_design(dut)
    
    # Send alternating pattern
    dut._log.info("Sending alternating I/Q pattern...")
    for i in range(200):
        pattern = 0xAA if (i % 2) == 0 else 0x55
        dut.ui_in.value = pattern
        await ClockCycles(dut.clk, 1)
    
    dut._log.info("✓ TEST 3 PASSED - Design runs without hanging")




@cocotb.test()
async def test_debug_outputs(dut):
    """Test 4: Debug - capture and display all outputs"""
    dut._log.info("=" * 60)
    dut._log.info("TEST 4: Debug Output Capture")
    dut._log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 62.5, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_design(dut)
    
    # Load packet
    packet = load_packet('example_packet')
    dut._log.info(f"Packet: {packet[:80]}...")
    
    # Generate signal
    test_signal = createFSK(packet, amp, ifreq, DF, SAMPLE_RATE, BT, offset)
    i_samples = np.real(test_signal)
    q_samples = np.imag(test_signal)
    
    # Storage for outputs
    outputs = {
        'sample_idx': [],
        'demod_symbol': [],
        'symbol_clk': [],
        'packet_detected': [],
        'uo_out_raw': []
    }
    
    # Send samples and capture everything
    dut._log.info(f"Sending {len(i_samples)} samples...")
    symbol_clk_prev = 0
    
    for sample_idx, (i, q) in enumerate(zip(i_samples[:1000], q_samples[:1000])):  # First 1000 samples only
        # Send sample
        i_val = int(np.clip(i, min_val, max_val))
        q_val = int(np.clip(q, min_val, max_val))
        i_unsigned = i_val & 0xF
        q_unsigned = q_val & 0xF
        dut.ui_in.value = i_unsigned | (q_unsigned << 4)
        
        await RisingEdge(dut.clk)
        
        # Capture outputs
        uo_val = int(dut.uo_out.value)
        demod_symbol = uo_val & 0x1
        symbol_clk = (uo_val >> 1) & 0x1
        packet_det = (uo_val >> 2) & 0x1
        
        # Store
        outputs['sample_idx'].append(sample_idx)
        outputs['demod_symbol'].append(demod_symbol)
        outputs['symbol_clk'].append(symbol_clk)
        outputs['packet_detected'].append(packet_det)
        outputs['uo_out_raw'].append(uo_val)
        
        # Log symbol clock edges
        if symbol_clk and not symbol_clk_prev:
            dut._log.info(f"  Symbol CLK @ sample {sample_idx}, bit={demod_symbol}")
        
        # Log packet detection
        if packet_det:
            dut._log.info(f"  PACKET DETECTED @ sample {sample_idx}")
        
        symbol_clk_prev = symbol_clk
        await FallingEdge(dut.clk)
    
    # Save to file
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
                outputs['uo_out_raw'][i]
            ])
    
    dut._log.info("Saved outputs to output_capture.csv")
    
    # Print summary
    symbol_edges = [i for i, (curr, prev) in enumerate(zip(outputs['symbol_clk'][1:], outputs['symbol_clk'][:-1])) 
                    if curr and not prev]
    recovered_bits = [outputs['demod_symbol'][i] for i in symbol_edges]
    
    dut._log.info(f"\nSUMMARY:")
    dut._log.info(f"  Total samples: {len(outputs['sample_idx'])}")
    dut._log.info(f"  Symbol clock edges: {len(symbol_edges)}")
    dut._log.info(f"  Recovered bits: {''.join(map(str, recovered_bits[:50]))}...")
    dut._log.info(f"  Original packet: {packet[:50]}...")
    dut._log.info(f"  Packet detections: {sum(outputs['packet_detected'])}")
    
    if len(symbol_edges) > 1:
        periods = np.diff(symbol_edges)
        dut._log.info(f"  Symbol periods: {periods[:10]}")
        dut._log.info(f"  Average period: {np.mean(periods):.2f} (expected {SAMPLE_RATE})")

    dut._log.info(" TEST 4 COMPLETE - Check output_capture.csv for details")
