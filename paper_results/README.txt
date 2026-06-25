This folder aims to be a self-contained storage for the data and plots shown in the SoRoMoX paper. The suggested programming language is Python. The proposed structure is the following:

- example_casestudy1/
	-> code/    -->	 contains the .py files that reads from the data/ folder to process and generate the plots
	-> data/    -->  contains the raw data in .csv, .npy, or .pkl format that are processed to generate the plots
	-> outputs/ -->  contains the plots and videos of the current case study, in .pdf and .mp4 format preferably

- example_casestudy2/
	-> code/
	-> data/
	-> outputs/

- ...

- final_outputs/    -->  contains the final plots and videos used in the paper, manually copy-pasted here when satisfied
		