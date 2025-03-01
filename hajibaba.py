"""
   PSO_automatic_data_saving.py

   
   Quantum Communication Group, Iranian Center for Quantum Technologies (ICQTs)
   modified: 05 March 2023

   Explain: Python code to run a particle swarm optimization (PSO) algorithm for minimizing
   the quantum bit error rate (QBER) of a quantum key distribution (QKD) system.

"""

import pyvisa
import serial
import numpy as np
import datetime as dt
import time
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import csv

# Connect Polarimeter
rm = pyvisa.ResourceManager()
polarimeter = rm.open_resource(rm.list_resources()[1])
print(polarimeter.query('*IDN?'))

# Initialization of Polarimeter
polarimeter.write('SENS:CALC 5;:INP:ROT:STAT 1;:INP:ROT:VEL 100') #(Mode, Motor On, Motor Speed in Hz)
print("Wavelength(m):",polarimeter.query('SENS:CORR:WAV?'))
print("Mode:",polarimeter.query('SENS:CALC?'))
print("Motor Speed(Hz):",polarimeter.query(':INP:ROT:VEL?'))

# Data Acquisition Begin
def Acquire_Data():
    polarimeter_data=list(map(float, polarimeter.query('SENS:DATA:LAT?').split(",")))
    #print(polarimeter_data)

    # Convert data to conventional units
    timestamp=polarimeter_data[1]
    mode=polarimeter_data[2]
    az=polarimeter_data[9]*180/np.pi # in degree
    ellip=polarimeter_data[10]*180/np.pi # in degree
    DOP=polarimeter_data[11]*100 # in %
    Power=polarimeter_data[12]*1e3 # in mW
    # Compute normalized Stokes parameters
    Psi=polarimeter_data[9];
    Chi=polarimeter_data[10];
    S1=np.cos(2*Psi)*np.cos(2*Chi) # normalized S1
    S2=np.sin(2*Psi)*np.cos(2*Chi) # normalized S2
    S3=np.sin(2*Chi) # normalized S3
    
    return S1, S2, S3, az, ellip # Output Parameter

# Connect Polarization Controller
Polarization_Controller = serial.Serial("COM20", 115200, timeout=1)

# Calculate QBER
QBERthreshold=0.05
def QBER(S1,ellip):
    return ((1+S1)/2)

# TEST
print("Azimuth=",Acquire_Data()[3])
print("Ellipticity=",Acquire_Data()[4])
print("S1=",Acquire_Data()[0])
print("S2=",Acquire_Data()[1])
print("S3=",Acquire_Data()[2])
print("QBER=",QBER(Acquire_Data()[0],Acquire_Data()[4])) #adjustable
#time.sleep(0.5)

def run_PSO():
    # PSO Algorithm
    begin_time=time.perf_counter()

    # Initialization
    max_particle_no=20
    iteration_max=20
    w=0.5
    c1=1
    c2=2
    X=np.empty([iteration_max,max_particle_no,4])
    Velocity=np.empty([iteration_max,max_particle_no,4])
    QBER_values=np.empty([iteration_max,max_particle_no])
    QBER_best=np.empty([max_particle_no])
    QBER_best_best=0.1 # 10% QBER
    Voltage_best=np.empty([iteration_max,max_particle_no,4])
    Voltage_best_best=[0,0,0,0]

    iteration=0

    for particle_no in range(max_particle_no):
        for dimension in range(4):
            X[iteration][particle_no][dimension]=np.random.randint(low=-5000, high=5001)
            Velocity[iteration][particle_no][dimension]=np.random.randint(low=-5000, high=5001)

        QBER_best[particle_no]=0.5 # 50% QBER

    flag=0
    # Optimization
    while iteration<(iteration_max-1):
        # Evaluate Cost function
        for particle_no in range(max_particle_no):
            # Applying Voltage
            if Polarization_Controller.isOpen():
                Polarization_Controller.write(("V1,"+str(int(X[iteration][particle_no][0]))+"\r\n").encode('ascii'))
                Polarization_Controller.write(("V2,"+str(int(X[iteration][particle_no][1]))+"\r\n").encode('ascii'))
                Polarization_Controller.write(("V3,"+str(int(X[iteration][particle_no][2]))+"\r\n").encode('ascii'))
                Polarization_Controller.write(("V4,"+str(int(X[iteration][particle_no][3]))+"\r\n").encode('ascii'))
                time.sleep(0.5)

            # Calculate QBER
            QBER_values[iteration][particle_no]=QBER(Acquire_Data()[0],Acquire_Data()[4])#adjustable
            writer.writerow([Acquire_Data()[3],Acquire_Data()[4],Acquire_Data()[0],Acquire_Data()[1],Acquire_Data()[2],QBER(Acquire_Data()[0],Acquire_Data()[4]), time.perf_counter()-start_time, int(X[iteration][particle_no][0]),int(X[iteration][particle_no][1]),int(X[iteration][particle_no][2]),int(X[iteration][particle_no][3])]) #adjustable
            
            # Checking cost function
            if QBER_values[iteration][particle_no]<=QBER_best[particle_no]:
                QBER_best[particle_no]=QBER_values[iteration][particle_no]
                Voltage_best[iteration][particle_no]=X[iteration][particle_no]

            if QBER_values[iteration][particle_no]<=QBER_best_best:
                QBER_best_best=QBER_values[iteration][particle_no]
                Voltage_best_best=X[iteration][particle_no]

            if QBER_best_best<QBERthreshold:
                flag=1
                # Applying Voltage
                if Polarization_Controller.isOpen():
                    Polarization_Controller.write(("V1,"+str(int(Voltage_best_best[0]))+"\r\n").encode('ascii'))
                    Polarization_Controller.write(("V2,"+str(int(Voltage_best_best[1]))+"\r\n").encode('ascii'))
                    Polarization_Controller.write(("V3,"+str(int(Voltage_best_best[2]))+"\r\n").encode('ascii'))
                    Polarization_Controller.write(("V4,"+str(int(Voltage_best_best[3]))+"\r\n").encode('ascii'))
                    time.sleep(0.5)

                Total_time=time.perf_counter()-begin_time
                # Output
                print("Iteration number=", iteration)
                print("Total Time(s)=", Total_time)
                print("Voltage Point(mV)=", Voltage_best_best)
                print("Minimum QBER=", QBER_best_best)
                break


            # Updating Velocities and positions
            for dimension in range(4):
                r1=np.random.randint(low=0, high=1)
                r2=np.random.randint(low=0, high=1)
                Velocity[iteration+1][particle_no][dimension]=w*Velocity[iteration][particle_no][dimension]+c1*r1*(Voltage_best[iteration][particle_no][dimension]-X[iteration][particle_no][dimension])+c2*r2*(Voltage_best_best[dimension]-X[iteration][particle_no][dimension])
                X[iteration+1][particle_no][dimension]=X[iteration][particle_no][dimension]+Velocity[iteration+1][particle_no][dimension]
                if X[iteration+1][particle_no][dimension]>5000:
                    X[iteration+1][particle_no][dimension]=5000
                if X[iteration+1][particle_no][dimension]<-5000:
                    X[iteration+1][particle_no][dimension]=-5000
            # Check if NaN
            try:
                X[iteration+1][particle_no]=[int(X[iteration+1][particle_no][0]),int(X[iteration+1][particle_no][1]),int(X[iteration+1][particle_no][2]),int(X[iteration+1][particle_no][3])]
            except:
                X[iteration+1][particle_no]=[0,0,0,0]
                print("error")
                break


        if flag==1:
            break

        print("Current QBER=",QBER_best_best)
        iteration+=1
        
# Plot QBER in Real-Time
# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

# Data Saving initialization
start_time=time.perf_counter()
timeofwrite=28800 #in seconds
file=open('PSO_automatic_data_full.csv','w')
writer=csv.writer(file)
writer.writerow(['Azimuth', 'Ellipticity', 'S1', 'S2', 'S3', 'QBER', 'time(s)', 'V1', 'V2', 'V3', 'V4'])

# This function is called periodically from FuncAnimation
def animate(i, xs, ys):
    
    if time.perf_counter()-start_time<timeofwrite:
        writer.writerow([Acquire_Data()[3],Acquire_Data()[4],Acquire_Data()[0],Acquire_Data()[1],Acquire_Data()[2],QBER(Acquire_Data()[0],Acquire_Data()[4]), time.perf_counter()-start_time, -1,-1,-1,-1]) #adjustable
    else:
        #print("Saving Finished.")
        file.close()   
    
    if QBER(Acquire_Data()[0],Acquire_Data()[4])>QBERthreshold: #adjustable
        run_PSO()
        
    # Read Y Parameter
    Y = QBER(Acquire_Data()[0],Acquire_Data()[4])*100 #adjustable

    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%S.%f'))
    ys.append(Y)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    ax.set_ylim([0, 100])
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('QBER value over Time')
    plt.ylabel("QBER (%)")

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=100)
plt.show()