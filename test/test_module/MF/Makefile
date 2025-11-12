# Makefile for Matched Filter Module Test
# Tests matched_filter.sv in isolation

# Simulator selection
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

# Source directory (go up 3 levels: MF -> test_module -> test -> project root)
SRC_DIR = $(PWD)/../../../src

# Test only the matched filter module
VERILOG_SOURCES = $(SRC_DIR)/matched_filter.sv
# VERILOG_SOURCES = $(SRC_DIR)/matched_filter_past.sv

# Module under test
TOPLEVEL = matched_filter

# Python test module (looks for test_matched_filter.py)
MODULE = test_matched_filter

# Build directory
SIM_BUILD = sim_build

# SystemVerilog support
COMPILE_ARGS += -g2012

# Allow access to include files if needed
COMPILE_ARGS += -I$(SRC_DIR)

# Include cocotb's makefile
include $(shell cocotb-config --makefiles)/Makefile.sim


