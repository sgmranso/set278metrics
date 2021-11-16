NATO SET 278 - WAMI truth-to-track metrics using CSV data
========

This directory contains code for computing multi-target metric using 
implementations with Stone-Soup and CSV truth/track data. Example
WAMI data has been provided to demonstrate the required format.


set278_read_csv.py
--------
Python reader file for CSV data and Stone-Soup components.


annotations.csv
--------

Truth data in the following format:

timestamp, seconds
	Time interval from initialisation.

track_id
	Unique truth path ID.

bbox_top_x
	Horizontal truth path position.

bbox_top_y
	Vertical truth path position.


tracks.csv
--------

Track data in the following format:

timestamp, seconds
	Time interval from initialisation.

track_id
	Unique track path ID.

bbox_top_x
	Horizontal track path position.

bbox_top_y
	Vertical track path position.
