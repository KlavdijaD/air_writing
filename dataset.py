import serial
import time

serial_port = 'COM4'
baud_rate = 115200
samples_total = 101
output = "Dataset\stevilka4.csv"

ser = serial.Serial(serial_port, baud_rate)
time.sleep(2) 

with open(output, 'w') as file:
    file.write("AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ, MagX, MagY, MagZ\n")
    i=0
    sample = 0
    while sample <= samples_total:#vsakič ko naleti na \n se samples poveča za 1, dokler ne pride do 100 oz 101 ker je prvi zapis corrupted(bounce-back)
        line = ser.readline().decode('utf-8').rstrip()
        if line:
            file.write(line + "\n") 
            i += 1
            #print(f"Sample {i+1} saved.")
            
        else:
            file.write("\n")
            sample += 1
            i=0
            print(f"Vzorec {sample-1} saved.")

ser.close()