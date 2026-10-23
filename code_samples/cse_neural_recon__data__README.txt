This README.txt file is designed to provide a brief overview of each folder and the files within the each folder.


1) Dataset name : CSE (C-slam in Service Environments)


2) Description 
In CSE dataset folder, there are 4 folders (hospital / office / warehouse / lifelong_sequence).

2.1) hospital / office / warehouse folders 
	- Those folders contain 2 sequences acquired in static and dynamic environments.
	- A total of 3 ground robots are utilized for each environment and are saved in separated bag files for each robot. 
	- Files : 
		- (static sequence) static_{env}_{robot_name).bag
		- (dynamic sequence) dynamic_{env}_{robot_name}.bag 
2.2) lifelong_sequence folder 
	- This folder contains 2 additional sequences for lifelong SLAM. 
	- We generated these sequences based on scene changes and reverse trajectory for a static office environment.
	- Files : 
		- (sequence 1) lifelong_static_office_origTraj_{robot_name}.bag 
		- (sequence 2) lifelong_static_office_reverseTraj_{robot_name}.bag 


3) Definition of words 
	1. robot_name : It matches the robot name (e.g., robot 1) described in main paper. 
	2. env : It means the environment name (e.g., hospital)
	3. static sequence : It means that the sequence obtained from each static environment. 
	4. dynamic sequence : It means that the sequence obtained from each dynamic evironment (w/ dynamic objects). 
	5. sequence 1 : scene changes + same trajectories as the "static office sequence".
	6. sequence 2 : scene changes + reverse trajectories of the "static office sequence".




%%%%%%%%%%%%%%%%% Record %%%%%%%%%%%%%%%%% 
This README.txt file was generated on 2024-04-11 by the CSE dataset authors. 