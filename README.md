# polarization_controlling
pure python APIs aim to handle polarization fluctuations in an optic setup

Photon polarisation plays a key role in most Quntum and classical photonic applications.
As an instance of its usage, in many quantum cryptography protocols, photon polarization is measured between sender and reciever sides (known as Alice and Bob) to make an agreement on a key.

However, due to some effects such as instruments imperfections, birefringence phenomena and thermal fluctuations, the polarization of photon could be changed as time passes through a fiber.
for many reasons, it maybe important to modify the polarisation to a certain value. to achieve this goal many instruments known as polarisation controllers have been created. these devices uses various methods, like manual dynamic ones to electrical ones that using Piezoelectrics, to adjust light polarisation into an arbitrary value by squeezing the fiber.

some new polarisation controllers are developed to track polarization and give feedback to the fiber online to achieve somehow "Endless Polarisation controlling".
(e.g https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-7-8259&id=282433 and https://digital-library.theiet.org/doi/abs/10.1049/el.2011.1522)

the main goal of development of these codes is also to give feedback from the fiber quickly and modify the polarisation by using a simple electrical polarisation controller.


to investigate the whole process, rather than using two laser devices, some beam splitters and polarization beam splitters, detectors, optical fibers and attenuators, we utilised an Oscilloscope to measure Polarization in a given fiber, An Electrical polarization controller to send different voltages to squeeze the fiber and a Polarimeter to acquire light polarizations directly.
former instruments are controlled by this code via a computer.

we implemented a prepare-and-mesure opticl setup to test the code efficiency. the light starts to emmit with a certain polarization in a one side and in the receiver side it should be the same as emitted. In Quantum cryptography context, any difference from the initial polarisation is described as Quantum Bit Error Rate (QBER).

in an ideal fiber the polarisation of emmited light should remain steady (with very minor fluctuations) over time. this stated means there is no Error rate in a Quantum Key distribution device ( which uses light polarisation as a quantum source of key agreement).

but in normal condition, polarization slightly drifts over time and QBER reaches a certain threshold.
one of the main parts of the code is to monitor the the polarization (and subsequently QBER) continuesly through an oscilloscope or polarimeter.

having crossed the threshold, the polarisation controller should begin to apply pressure on fiber to subside the increased QBER.

the key role in this part is to develop an optimiser code to find the optimal value of applying voltages. this lead to decrease time of correction process.

in our experiment we developed this code to deal with these devices:

Polarimeter: ThorLabs PAX1000IR2
Polarisation controller: OZ Optics 4 channels EPC Polarisation controller with USB Driver
Oscilloscopes: OWON VDS6047 PC Oscilloscope
	       RIGOL DS 6104 Oscilloscope

Moreover we have developed Particle Swarm Optimisation (PSO) and Simulated Annealing (SA) algorithms to optimise the error correction time in high QBER regimes.


the code is designed to easily implement and develop any other instruments or optimisers.
Beside's I think APIs developed for instruments might me useful for other intentions if anyone wants to use with those devices.


I hope this code would be useful for even one person who wants to use it scientific purposes.
