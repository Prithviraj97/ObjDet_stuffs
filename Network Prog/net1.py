from scapy.all import *

# Define a function to handle each packet
def packet_sniffer(packet):
    # Print the packet summary
    print(packet.summary())

    # Check if the packet is a TCP packet
    if packet.haslayer(TCP):
        # Print the TCP packet details
        print("TCP Packet:")
        print("Source IP:", packet[IP].src)
        print("Destination IP:", packet[IP].dst)
        print("Source Port:", packet[TCP].sport)
        print("Destination Port:", packet[TCP].dport)

    # Check if the packet is a UDP packet
    elif packet.haslayer(UDP):
        # Print the UDP packet details
        print("UDP Packet:")
        print("Source IP:", packet[IP].src)
        print("Destination IP:", packet[IP].dst)
        print("Source Port:", packet[UDP].sport)
        print("Destination Port:", packet[UDP].dport)

# Start sniffing packets
sniff(prn=packet_sniffer, store=0)