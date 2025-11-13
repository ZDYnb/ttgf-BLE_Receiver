import random
import os
from pathlib import Path


packets_path = os.path.join(Path(__file__).parent.parent.parent, "Packet Strings")


def bitstream_gen(numPackets, totalBits):
    with open(os.path.join(packets_path, "connectable.txt")) as f:
        packet_bits = f.read()

    bits = totalBits - (numPackets * len(packet_bits))
    bitstream = bin(random.getrandbits(bits))[2:]
    bitstream = '0' * (bits - len(bitstream)) + bitstream

    startIndexes = sorted(random.choices(range(numPackets), k = numPackets))
    indexes = []
    
    for i, base_ix in enumerate(startIndexes):
        index = i * len(packet_bits) + base_ix
        bitstream = bitstream[:index] + packet_bits + bitstream[index:]
        indexes.append(index)

    return bitstream, indexes


if __name__ == "__main__":
    random.seed()

    max_packets = 2000
    numPackets = random.randrange(max_packets + 1)

    bitstream, indexes = bitstream_gen(numPackets, 10 ** 6)

    print(numPackets)
    print(sorted(indexes))
    print(len(bitstream))

    with open(os.path.join(packets_path, "random_bits_10kb.txt"), "w") as f:
        f.write(bitstream)

