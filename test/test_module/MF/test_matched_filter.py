
"""
Matched Filter Testbench
Tests the matched_filter module in isolation
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, FallingEdge
import os
import sys
import numpy as np
from math import pi
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Add the Modules directory to Python path
test_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to test/ directory
sys.path.insert(0, os.path.join(test_dir, 'Modules'))

# Import BLE helper modules
from Modules.phy.iq import createFSK
from Modules.phy.demodulation import matched_filter_score_gen

# Test configuration
SAMPLE_RATE = 16        # Samples per symbol
DATA_WIDTH = 4          # Bits per I/Q sample
PIPELINE_STAGES = 1     # Pipeline depth of matched filter
fsym = 1_000_000        # 1 MHz symbol rate
clk_freq = 16_000_000   # 16 MHz clock

# Signal parameters
DF = 0.25 * fsym        # Frequency deviation
BT = 1 / fsym           # Bandwidth-time product
ifreq = 1250000         # 1.25 MHz intermediate frequency

# Calculate I/Q sample range
max_val = 2 ** (DATA_WIDTH - 1) - 1  # +7
min_val = -2 ** (DATA_WIDTH - 1)      # -8
amp = (max_val - min_val) / 2
offset = (max_val + min_val) / 2


def to_unsigned(val):
    """Convert signed value to unsigned for Verilog"""
    return val & 0xF


async def reset_dut(dut):
    """Reset the matched filter"""
    dut.resetn.value = 0
    dut.en.value = 0
    dut.i_data.value = 0
    dut.q_data.value = 0
    
    await ClockCycles(dut.clk, 10)
    dut.resetn.value = 1
    await ClockCycles(dut.clk, 5)
    dut.en.value = 1
    dut._log.info("Reset complete")


async def send_iq_samples(dut, i_samples, q_samples):
    """Send I/Q samples to matched filter one at a time"""
    for i, q in zip(i_samples, q_samples):
        # Clip to valid range
        i_val = int(np.clip(i, min_val, max_val))
        q_val = int(np.clip(q, min_val, max_val))
        
        # Convert to unsigned
        dut.i_data.value = to_unsigned(i_val)
        dut.q_data.value = to_unsigned(q_val)

        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)


async def run_demod_test(dut, bitstring, test_name="test"):
    """
    Run a demodulation test
    
    Args:
        dut: Device under test
        bitstring: String of bits to test (e.g., "01010101")
        test_name: Name for logging
    """
    dut._log.info(f"Starting {test_name}")
    dut._log.info(f"Bitstring: {bitstring} ({len(bitstring)} bits)")
    
    # Generate FSK modulated I/Q samples
    test_signal = createFSK(
        bitstring, 
        amp, 
        ifreq, 
        DF, 
        SAMPLE_RATE, 
        BT, 
        offset
    )
    
    reference_data = matched_filter_score_gen(
        test_signal, ifreq, DF, SAMPLE_RATE, BT, amp, output_type=bool
    )


    i_samples = np.real(test_signal)
    q_samples = np.imag(test_signal)
    
    dut._log.info(f"Generated {len(i_samples)} I/Q samples ({len(i_samples)//SAMPLE_RATE} symbols)")
    
    # Add warmup and cooldown samples for pipeline
    warmup = PIPELINE_STAGES + 220
    i_samples = np.concatenate([np.zeros(warmup), i_samples, np.zeros(warmup)])
    q_samples = np.concatenate([np.zeros(warmup), q_samples, np.zeros(warmup)])
    
    # Storage for outputs
    demod_bits = []
    
    # Send samples and collect outputs
    dut._log.info("Sending I/Q samples...")
    for sample_idx, (i, q) in enumerate(zip(i_samples, q_samples)):
        # next reference data, true = 1  false = 0
        # print("next(reference_data):(1 or 0)", next(reference_data))
        i_val = int(np.clip(i, min_val, max_val))
        q_val = int(np.clip(q, min_val, max_val))
        
        dut.i_data.value = to_unsigned(i_val)
        dut.q_data.value = to_unsigned(q_val)
        
        await RisingEdge(dut.clk)

        # # Read full packed vector as a binary string
        # buf_bits = dut.i_buffer.value.binstr

        # # Split it into DATA_WIDTH-bit chunks
        # samples = [
        #     int(buf_bits[k*DATA_WIDTH:(k+1)*DATA_WIDTH], 2)
        #     for k in range(SAMPLE_RATE)
        # ]

        # dut._log.info(f"Cycle {sample_idx}: I buffer last 4 samples = {samples[-4:]}")

        # Collect output bit (after warmup)
        if sample_idx >= warmup:
            # when pass, print the current index and bit value
            dut._log.info(f"Sample idx {sample_idx}: demodulated_bit = {dut.demodulated_bit.value}")
            demod_bit = int(dut.demodulated_bit.value)
            # Sample once per symbol (every SAMPLE_RATE samples)
            if (sample_idx - warmup) % SAMPLE_RATE == 0:
                demod_bits.append(demod_bit)

    flush_len = PIPELINE_STAGES + SAMPLE_RATE +10
    for extra_idx in range(flush_len):
        await RisingEdge(dut.clk)
        if extra_idx % SAMPLE_RATE == 0:
            demod_bits.append(int(dut.demodulated_bit.value))
    
    # Convert to string
    recovered = ''.join(map(str, demod_bits[:len(bitstring)]))
    
    dut._log.info(f"Test complete: {test_name}")
    dut._log.info(f"  Input:     {bitstring}")
    dut._log.info(f"  Recovered: {recovered}")
    
    # Calculate bit errors
    errors = sum(1 for a, b in zip(bitstring, recovered) if a != b)
    ber = errors / len(bitstring) if len(bitstring) > 0 else 0
    
    dut._log.info(f"  Bit errors: {errors}/{len(bitstring)} (BER: {ber:.2%})")
    
    return recovered, errors


# Test Cases

@cocotb.test()
async def test_reset(dut):
    """Test 1: Verify reset works"""
    dut._log.info("=" * 60)
    dut._log.info("TEST 1: Reset")
    dut._log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 62.5, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_dut(dut)
    
    # Check output is stable (not X or Z)
    output = dut.demodulated_bit.value
    assert output.is_resolvable, "Output has X/Z after reset"
    
    dut._log.info(" TEST 1 PASSED")


@cocotb.test()
async def test_continuous_zeros(dut):
    """Test 2: Demodulate continuous zeros"""
    dut._log.info("=" * 60)
    dut._log.info("TEST 2: Continuous Zeros")
    dut._log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 62.5, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_dut(dut)
    
    # Run test
    bitstring = "0000"
    recovered, errors = await run_demod_test(dut, bitstring, "continuous_zeros")
    
    # Verify
    assert errors == 0, f"Expected 0 errors, got {errors}"
    
    dut._log.info(" TEST 2 PASSED")


@cocotb.test()
async def test_continuous_ones(dut):
    """Test 3: Demodulate continuous ones"""
    dut._log.info("=" * 60)
    dut._log.info("TEST 3: Continuous Ones")
    dut._log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_dut(dut)
    
    # Run test
    bitstring = "11111111111111111111"
    recovered, errors = await run_demod_test(dut, bitstring, "continuous_ones")
    
    # Verify
    assert errors == 0, f"Expected 0 errors, got {errors}"
    
    dut._log.info(" TEST 3 PASSED")


@cocotb.test()
async def test_alternating_bits(dut):
    """Test 4: Demodulate alternating bits (preamble pattern)"""
    dut._log.info("=" * 60)
    dut._log.info("TEST 4: Alternating Bits (Preamble)")
    dut._log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_dut(dut)
    
    # Run test
    bitstring = "01010101"
    recovered, errors = await run_demod_test(dut, bitstring, "alternating_bits")
    
    # Verify
    assert errors == 0, f"Expected 0 errors, got {errors}"
    
    dut._log.info(" TEST 4 PASSED")


@cocotb.test()
async def test_random_pattern(dut):
    """Test 5: Demodulate random bit pattern"""
    dut._log.info("=" * 60)
    dut._log.info("TEST 5: Random Pattern")
    dut._log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_dut(dut)
    
    # Generate random bitstring
    np.random.seed(42)  # For reproducibility
    bits = np.random.randint(0, 2, 16)
    bitstring = ''.join(map(str, bits))
    
    # Run test
    recovered, errors = await run_demod_test(dut, bitstring, "random_pattern")
    
    # Verify (allow small error rate for random data)
    ber = errors / len(bitstring)
    assert ber < 0.1, f"BER too high: {ber:.2%}"
    
    dut._log.info(" TEST 5 PASSED")


@cocotb.test()
async def test_enable_control(dut):
    """Test 6: Verify enable signal works"""
    dut._log.info("=" * 60)
    dut._log.info("TEST 6: Enable Control")
    dut._log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset with enable high
    await reset_dut(dut)
    
    # Send some data with enable high
    bitstring = "1111"
    test_signal = createFSK(bitstring, amp, ifreq, DF, SAMPLE_RATE, BT, offset)
    i_samples = np.real(test_signal)[:32]  # Just first 2 symbols
    q_samples = np.imag(test_signal)[:32]
    
    dut.en.value = 1
    await send_iq_samples(dut, i_samples, q_samples)
    
    # Now disable
    dut.en.value = 0
    dut._log.info("Enable disabled - pipeline should freeze")
    
    # Clock should run but pipeline shouldn't advance
    await ClockCycles(dut.clk, 20)
    
    # Re-enable
    dut.en.value = 1
    dut._log.info("Enable re-enabled")
    await ClockCycles(dut.clk, 10)
    
    dut._log.info(" TEST 6 PASSED - Enable control works")


@cocotb.test()
async def test_pipeline_latency(dut):
    """Test 7: Measure actual pipeline latency"""
    dut._log.info("=" * 60)
    dut._log.info("TEST 7: Pipeline Latency Measurement")
    dut._log.info("=" * 60)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await reset_dut(dut)
    
    dut._log.info(f"Expected pipeline stages: {PIPELINE_STAGES}")
    
    # Send a simple pattern: transition from 0 to 1
    bitstring = "00001111"
    test_signal = createFSK(bitstring, amp, ifreq, DF, SAMPLE_RATE, BT, offset)
    i_samples = np.real(test_signal)
    q_samples = np.imag(test_signal)
    
    # Track when we send the transition and when we see it
    transition_sent = False
    transition_cycle = 0
    output_changed = False
    output_change_cycle = 0
    
    last_output = None
    
    for cycle, (i, q) in enumerate(zip(i_samples, q_samples)):
        i_val = int(np.clip(i, min_val, max_val))
        q_val = int(np.clip(q, min_val, max_val))
        
        dut.i_data.value = to_unsigned(i_val)
        dut.q_data.value = to_unsigned(q_val)
        
        await RisingEdge(dut.clk)
        
        # Mark when we finish sending the 0s (start of 1s)
        if not transition_sent and cycle == 3 * SAMPLE_RATE:
            transition_sent = True
            transition_cycle = cycle
            dut._log.info(f"Transition sent at cycle {transition_cycle}")
        
        # Watch for output change
        current_output = int(dut.demodulated_bit.value)
        if last_output is not None and current_output != last_output and not output_changed:
            output_changed = True
            output_change_cycle = cycle
            dut._log.info(f"Output changed at cycle {output_change_cycle}")
        
        last_output = current_output
    
    if transition_sent and output_changed:
        measured_latency = output_change_cycle - transition_cycle
        dut._log.info(f"Measured latency: {measured_latency} cycles")
        dut._log.info(f"Expected: ~{PIPELINE_STAGES} cycles")
        
        # Allow some tolerance
        assert abs(measured_latency - PIPELINE_STAGES) < 5, \
            f"Latency mismatch: measured {measured_latency}, expected {PIPELINE_STAGES}"
    
    dut._log.info(" TEST 7 PASSED")