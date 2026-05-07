import asyncio
import csv
from datetime import datetime
from bleak import BleakClient, BleakScanner
import time

# --- CONFIGURATION ---
TARGET_NAMES = ["ESP32_GAIT", "ESP32_GAIT2"]
SERVICE_UUID = "12345678-1234-1234-1234-1234567890ab"
CHAR_UUID    = "abcd1234-1234-1234-1234-abcdefabcdef"

# CSV and State management
files = {}
writers = {}
stop_event = asyncio.Event()

def notification_handler(sender, data, name):
    """Parses data and adds a PC-side timestamp for sync."""
    try:
        decoded = data.decode('utf-8').strip()
        if decoded.startswith('$'):
            line = decoded[1:] 
            fields = line.split(',')
            if len(fields) == 9:
                pc_time = time.time() # Sync timestamp
                values = [pc_time] + [float(x) for x in fields]
                writers[name].writerow(values)
    except Exception as e:
        print(f"[{name}] Parse error: {e}")

async def scan_for_esp32():
    print("Searching for gait sensors (ESP32_GAIT & ESP32_GAIT2)...")
    devices = await BleakScanner.discover(timeout=5.0)
    found = {}
    for d in devices:
        if d.name in TARGET_NAMES:
            found[d.name] = d.address
            print(f"  [FOUND] {d.name} at {d.address}")
    return found

async def connect_and_listen(address, name):
    print(f"[{name}] Attempting to connect...")
    try:
        async with BleakClient(address, timeout=15.0) as client:
            print(f"[{name}] Connected!")
            
            # Cross-platform MTU check
            if hasattr(client, "exchange_mtu"):
                try:
                    await client.exchange_mtu(512)
                    print(f"[{name}] MTU set to 512")
                except Exception as e:
                    print(f"[{name}] MTU exchange failed: {e}")

            # Register the global notification handler with the 'name' argument
            await client.start_notify(
                CHAR_UUID, 
                lambda sender, data: notification_handler(sender, data, name)
            )
            
            while not stop_event.is_set():
                if not client.is_connected:
                    break
                await asyncio.sleep(0.01)

            await client.stop_notify(CHAR_UUID)
    except Exception as e:
        print(f"[{name}] Connection error: {e}")
    finally:
        print(f"[{name}] Disconnected.")

async def main():
    found_devices = await scan_for_esp32()
    missing = [n for n in TARGET_NAMES if n not in found_devices]
    
    if missing:
        print(f"\nERROR: Could not find: {missing}")
        return

    # Prepare CSV Files with the new PC_TIME header
    header = ["pc_time", "time", "fsr1", "fsr2", "ax", "ay", "az", "gx", "gy", "gz"]
    for name in TARGET_NAMES:
        filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        f = open(filename, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(header)
        files[name] = f
        writers[name] = writer
        print(f"Logging {name} to {filename}")

    print("\nStarting Data Collection. Press Ctrl+C to STOP.\n")
    try:
        # Run both connections until Ctrl+C
        await asyncio.gather(
            *(connect_and_listen(found_devices[name], name) for name in TARGET_NAMES)
        )
    finally:
        for f in files.values():
            f.close()
        print("\nAll data saved to CSV files.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping...")
        stop_event.set()