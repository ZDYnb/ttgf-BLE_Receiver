import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
from helpers import bin2hex, hex2bin, reverse_string, calcCRC, whiten_fullPacket 
from ble_packet_decode import packet_decode


# Generates a BLE Packet based on given data


def packet_gen(_access_address, adv_type, _adv_address, data):
    # Convert access address to binary and reverse
    access_address = reverse_string(hex2bin(_access_address))

    # Calculate preamble based on access address
    preamble = hex2bin('55') if access_address[0] == '0' else hex2bin('aa')

    # Calculate advertising address
    adv_address = reverse_string(hex2bin(_adv_address))

    # Calculate payload
    payload = ''

    # Only contains 'useful' GAP types (for now)
    GAP = {'FLAGS': '01', '128BIT_SERVICE_UUID_COMPLETE': '07', 'SHORT_LOCAL_NAME': '08', 'COMPLETE_LOCAL_NAME': '09', 'MANUFACTURER_SPECIFIC_DATA': 'ff'}
    for type, val in data:
        if 'NAME' in type:
            if isinstance(val, int):
                val = hex(val)[2:]
                
                if len(val) % 2 == 1:
                    val = '0' + val
            else:
                val = ''.join(hex(ord(v))[2:] for v in val)

        length = hex(len(val) // 2 + 1)[2:]
        if len(length) % 2 == 1:
            length = '0' + length

        payload += reverse_string(hex2bin(length))
        payload += reverse_string(hex2bin(GAP[type]))

        for i in range(0, len(val), 2):
            byte = val[i:i + 2]
            payload += reverse_string(hex2bin(byte))

    if len(payload) / 8 > 31:
        raise ValueError("Error: Payload too big (at most 31 bytes in length)")

    # Calculate header
    header = ''

    pdu_types = {'ADV_IND': '0000', 'ADV_DIRECT_IND': '0001', 'ADV_NONCONN_IND': '0010', 'SCAN_REQ': '0011', 'SCAN_RSP': '0100', 'CONNECT_REQ': '0101', 'ADV_SCAN_IND': '0110'}
    rfu = '00'
    txadd = '0'
    rxadd = '0'

    header += reverse_string(pdu_types[adv_type])
    header += rfu
    header += txadd
    header += rxadd

    length = hex((len(adv_address) + len(payload)) // 8)[2:]
    if len(length) % 2 == 1:
        length = '0' + length
    
    header += reverse_string(hex2bin(length))

    # Calcualte crc
    crc = ''.join(calcCRC(header + adv_address + payload))

    output = preamble + access_address + header + adv_address + payload + crc

    return bin2hex(output)


if __name__ == "__main__":
    ble_packet = packet_gen('8e89bed6', 'ADV_NONCONN_IND', '90d7ebb19299', [['FLAGS', '06'], ['COMPLETE_LOCAL_NAME', 'SCUM3']])
    ble_packet = packet_gen('8e89bed6', 'ADV_IND', '90d7ebb19299', [['FLAGS', '02'], ['COMPLETE_LOCAL_NAME', 'SCUM3']])
    ble_packet = packet_gen('8e89bed6', 'SCAN_RSP', '90d7ebb19299', [['FLAGS', '06'], ['COMPLETE_LOCAL_NAME', 'SCUM3']])
    
    print(f"Full BLE Packet: 0x{ble_packet} ({len(ble_packet) // 2} bytes)\n")
    # with open("Text Files/testpacket.txt", "w") as f:
    #     f.write(f"{ble_packet}")

    channel = int(input("Channel: "))
    ble_packet = whiten_fullPacket(ble_packet, channel)
    print(f"Whitened Packet: 0x{ble_packet} ({len(ble_packet) // 2} bytes)\n")

    print(f"\nRaw PDU: 0x{packet_decode(whiten_fullPacket(ble_packet, channel))}\n")

    # if input("Store whitened packet (y/n)? ") == 'y':
    #     with open("Text Files/testpacket.txt", "w") as f:
    #         f.write(f"{ble_packet}")