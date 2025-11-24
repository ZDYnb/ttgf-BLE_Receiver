<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->
## How it works

This project implements a Bluetooth Low Energy (BLE) digital baseband receiver designed for the SCuM (Single-Chip µ-Mote) platform (https://crystalfree.atlassian.net/wiki/spaces/SCUM/overview). The receiver takes I/Q samples from the SCuM RF front-end (or any compatible I/Q-sample interface), processes incoming BLE signals in real time, and performs packet decoding to extract BLE packets.

Core processing stages:
- Matched filtering for GFSK demodulation and bit extraction
- Clock and data recovery for symbol-timing synchronization
- Preamble-detection module for identifying the start of a received BLE packet
- Packet-sniffer module that performs bit de-whitening, CRC checking, and detection of complete BLE packets

## How to test

Connect the SCuM chip’s I/Q sampling outputs to the BLE digital baseband receiver. Configure the SCuM RF front-end to receive BLE packets, and then observe the decoded packet data on a computer through the Tiny Tapeout chip’s output interface.

## External hardware

- SCuM Chip – Serves as the RF front-end and BLE transmitter/receiver interface.
- Digital Discovery – Used for signal probing, debugging, and verification.
- Computer – Handles serial communication, visualization, and data logging.