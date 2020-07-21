# Foraging behavior diagnostics
Purpose: taking data from PharaGlow and analyzing the velocity and pumping of worm strains to get a sense of what the general trend of the data is. There is also a tool that makes counting pumping manually more streamlined, in case manual data is needed as a base for checking automatic detection.

## Installation

The environment from PharaGlow is necessary for the Notebooks in this repository to work, so make sure the pumping environment is active before use (source activate pumping).

## Post PharaGlow analysis

There are two Notebook options depending on if the recording condition has a lawn or not.

### With lawn:

1. Download the Post-pharaglow_V&P_analysis(easysave).ipynb file.

2. Change file directories, variables, etc. where (user input required) is stated in the heading - there should be instructions on what to change as comments.

	a) The first section with changeable variables is at the very top of the script under "Changing names" (very self-explanatory).
	
	b) The second section is under "Reading and Graphing". Here the user needs to copy and paste a section for each new condition and change the index in the order the conditions are added in the directories list.

3. Wait for all of your data to be loaded into the Notebook and analyzed. This data will be saved by default (unless save = False) so that this step only needs to be performed once.

4. Next the pumping rate, velocity and bacterial fluorescence around the nose tip are plotted. Here, the user is given the option of either having an average of all trajectories or each trajectory plotted. The figures should also save themselves automatically unless save == 'No'.

### Without lawn:

Download the Post-pharaglow_V&P_analysis(no_lawns).ipynb file.

Virtually the same as the above, except that the data is represented as histograms and there is no fluorescence plot.

## Movie Maker

Download the movieMaker.ipynb file.

This file creates an interactive "movie" of a recording within the Notebook so that the pharynx during one worm's trajectory is tracked for the user to see, and a button allows you to keep track of pumping events in a list. Once you're done counting, simply print the list and copy the values over to a text file.
