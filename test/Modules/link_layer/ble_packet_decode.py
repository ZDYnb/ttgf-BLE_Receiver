import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
from helpers import bin2hex, hex2bin, reverse_string, calcCRC, whiten_fullPacket

# Decodes a packet
# Must be the packet only, no extra bits before/after


def packet_decode(packet, channel=None, verbose=True):
    if all(c in '01' for c in packet):
        packet = bin2hex(packet)

    if channel is not None:
        packet = whiten_fullPacket(packet, channel)

    # Obtain preamble, access address, header, advertising address, payload, and CRC
    preamble = packet[:2]
    access_address = packet[2:10]
    header = packet[10:14]
    adv_address = packet[14:26]
    payload = packet[26:-6]
    crc = packet[-6:]


    # Check preamble and access address
    packet_start = hex2bin(packet)

    for i in range(8):
        if packet_start[i] == packet_start[i + 1]:
            raise ValueError("Preamble incorrect for access address")
    
    if verbose:
        print(f"Preamble: 0x{preamble}")
        print(f"Access Address: 0x{bin2hex(reverse_string(hex2bin(access_address)))}")

    # Check header
    if verbose:
        print("Header:")

        pdu_types = {'0000' :'ADV_IND', '0001': 'ADV_DIRECT_IND','0010': 'ADV_NONCONN_IND', '0011': 'SCAN_REQ', '0100': 'SCAN_RSP','0101': 'CONNECT_REQ', '0110': 'ADV_SCAN_IND'}
        print(f"\tPDU Type: {pdu_types[reverse_string(hex2bin(header[0]))]}")
        txadd, rxadd = hex2bin(header[1])[-2:]
        print(f"\tTx Address: {'Public' if txadd == '0' else 'Random'}\n\tRx Address: {'Public' if rxadd == '0' else 'Random'}")
        payloadLen = int(reverse_string(hex2bin(header[-2:])), 2)
        print(f"\tPayload Length: {payloadLen} bytes")

    # Check Payload
    if verbose:
        print(f"Payload:")

        payloadLen -= 6
        blocks = [0]
        while blocks[-1] < payloadLen:
            blockLen = int(reverse_string(hex2bin(payload[blocks[-1] * 2 :blocks[-1] * 2 + 2])), 2)
            blocks.append(blocks[-1] + blockLen + 1)
            

        if blocks[-1] != payloadLen:
            raise ValueError("Payload length is longer than expected")
        
        blocks.pop(-1)

        print(f"\tAdvertiser Address: 0x{bin2hex(reverse_string(hex2bin(adv_address)))}")
        print(f"\t{len(blocks)} block(s)")

        GAP = {'01': 'FLAGS', '07': '128BIT_SERVICE_UUID_COMPLETE', '08': 'SHORT_LOCAL_NAME', '09': 'COMPLETE_LOCAL_NAME', '02': '16BIT_SERVICE_UUID_MORE_AVAILABLE', '03': '16BIT_SERVICE_UUID_COMPLETE', '0a': 'TX_POWER_LEVEL',  '12': 'SLAVE_CONNECTION_INTERVAL_RANGE', '1a': 'ADVERTISING_INTERVAL', 'ff': 'MANUFACTURER_SPECIFIC_DATA'}
        for i in range(len(blocks)):
            print(f"\tBlock {i + 1}:")

            blockType = GAP[bin2hex(reverse_string(hex2bin(payload[(blocks[i] + 1) * 2 : (blocks[i] + 1) * 2 + 2])))]
            blockBytes = [bin2hex(reverse_string(hex2bin(payload[(k) * 2 : (k + 1) * 2]))) for k in range(blocks[i] + 2, payloadLen if i == len(blocks) - 1 else blocks[i + 1])]
            print(f"\t\tBlock Type: {blockType}")

            if blockType == 'FLAGS':
                flagVal = int(blockBytes[0], 16)

                if flagVal & 0x8:
                    print("\t\tSimultaneous LE & BR/EDR Supported")
                
                if flagVal & 0x4:
                    print("\t\tBR/EDR Not Supported")

                if flagVal & 0x02:
                    print("\t\tLE General Discoverable Mode")

                if flagVal & 0x01:
                    print("\t\tLE Limited Discoverable Mode")
            elif 'NAME' in blockType:
                print(f"\t\tName: {''.join(chr(int(x, 16)) for x in blockBytes)}")
            elif blockType == 'MANUFACTURER_SPECIFIC_DATA':
                print(f"\t\tCompany ID: 0x{''.join(blockBytes[1:-len(blockBytes) - 1:-1])}")
                print(f"\t\tData: 0x{''.join(blockBytes[2:])}")
            else:
                print(f"\t\t0x{''.join(blockBytes)}")

    # Check crc
    if int(''.join(calcCRC(hex2bin(header + adv_address + payload + crc)))) != 0:
        raise ValueError("CRC Check Failed")
    
    if verbose:
        print(f"CRC: 0x{crc}")
        print(f"\tCRC Check Passed")

    # Assemble raw PDU
    rawPDU = bin2hex(reverse_string(hex2bin(header[0]))) + header[1] + bin2hex(reverse_string(hex2bin(header[-2:])))

    for i in range(0, len(adv_address), 2):
        rawPDU += bin2hex(reverse_string(hex2bin(adv_address[i:i+2])))

    for i in range(0, len(payload), 2):
        rawPDU += bin2hex(reverse_string(hex2bin(payload[i:i+2])))

    return rawPDU


if __name__ == "__main__":
    with open("Text Files/testpacket.txt") as f:
        packet = f.readline()

    print(f"\nRaw PDU: 0x{packet_decode(packet, 37, True)}")