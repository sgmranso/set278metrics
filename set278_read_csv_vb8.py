#!/usr/bin/env python
# coding: utf-8

"""
NATO SET-278 : read_csv
===============
This Python script reads data from annotation and track CSV files
and generates corresponding metrics using Stone-Soup class methods.
"""

# %%
# imports
import datetime
import numpy as np
from ordered_set import OrderedSet
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas import read_csv as pd_read_csv
import pathlib as pl
from stonesoup.types.state import StateVector
from stonesoup.types.state import State
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.types.track import Track

# set csv data sets
annotations_csv_path = input("Enter path for annotations.csv: ")
tracks_csv_path = input("Enter path for tracks.csv: ")

# use pandas to create dataframes from 'annotations.csv' and 'tracks.csv'
print('Loading truths CSV file...')
annotation_df = pd_read_csv(annotations_csv_path)
print('Loading tracks CSV file...')
track_df = pd_read_csv(tracks_csv_path)

# set current time as initial timestamp
initial_timestamp = datetime.datetime.now()

# find all time_steps in data for tracks and truths inclusive
# rounding ensures matching truth to track
print('Collating timestamps for truths and tracks...')
n_round = 2
truth_time_steps = annotation_df.timestamp  # truth timestamps
truth_time_steps = [round(time_step, n_round) for time_step in truth_time_steps]  # convert to list
track_time_steps = track_df.timestamp  # track timestamps
track_time_steps = [round(time_step, n_round) for time_step in track_time_steps]  # convert to list
time_steps = [*truth_time_steps, *track_time_steps]  # add all time_steps to list
time_steps = list(dict.fromkeys(time_steps))  # all time_steps with duplicates removed
n_time_steps = int(len(time_steps))  # count unique time_steps

# timestamps
timestamps = []
for time_step in time_steps:
    timestamp = initial_timestamp + datetime.timedelta(
        seconds=time_step)
    timestamps.append(timestamp)
max_timestamp = max(timestamps)

# Extract annotation positions by groundtruth ID
print('Extracting truth states...')
annotation_ids = annotation_df.track_id  # all truth IDs
annotation_ids = annotation_ids.drop_duplicates()  # remove non-unique truth IDs
annotation_ids = [track_id for track_id in annotation_ids]
states = [[] for annotation_id in annotation_ids]  # set of sets storing states on truth ID
for _, annotation_set in annotation_df.iterrows():  # loop on each row in the csv dataframe
    annotation_id = annotation_set.track_id
    annotation_idx = annotation_ids.index(annotation_id)  # states set index
    state_vector = StateVector(
        [[annotation_set.bbox_x],
         [annotation_set.bbox_y]])
    time_step = datetime.timedelta(
        seconds=round(annotation_set.timestamp, n_round))  # time_step in seconds
    timestamp = initial_timestamp + time_step  # add time_step to initial time
    state = GroundTruthState(
        state_vector=state_vector,
        timestamp=timestamp)
    states[annotation_idx].append(state)
    if timestamp > max_timestamp:
        break

# assign states by groundtruth
print('Creating groundtruth paths and assigning states...')
groundtruth_paths = []
for annotation_id in annotation_ids:
    annotation_idx = annotation_ids.index(annotation_id)
    groundtruth_paths.append(GroundTruthPath(
        id=annotation_id,
        states=states[annotation_idx]))

# Extract track positions by track ID
print('Extracting track states...')
track_ids = track_df.track_id  # all track IDs
track_ids = track_ids.drop_duplicates()  # remove non-unique track IDs
track_ids = [track_id for track_id in track_ids]
states = [[] for track_id in track_ids]  # set of sets storing states on track ID
for _, track_set in track_df.iterrows():
    track_id = track_set.track_id
    track_idx = track_ids.index(track_id)
    state_vector = StateVector(
        [[track_set.bbox_x],
         [track_set.bbox_y]])
    time_step = datetime.timedelta(
        seconds=round(track_set.timestamp, n_round))
    timestamp = initial_timestamp + time_step
    state = State(
        state_vector=state_vector,
        timestamp=timestamp)
    states[track_idx].append(state)
    if timestamp > max_timestamp:
        break

# assign states by track
print('Creating tracks and assigning states...')
tracks = []
for track_id in track_ids:
    track_idx = track_ids.index(track_id)
    tracks.append(Track(
        id=track_id,
        states=states[track_idx]))

# assign truths and tracks by timestamp
groundtruth_paths_by_timestamp = [OrderedSet() for timestamp in timestamps]
tracks_by_timestamp = [OrderedSet() for timestamp in timestamps]
for timestamp in timestamps:
    time_idx = timestamps.index(timestamp)
    for groundtruth_path in groundtruth_paths:
        for state in groundtruth_path.states:
            if state.timestamp > timestamp:
                break
            if state.timestamp == timestamp:
                groundtruth_paths_by_timestamp[time_idx].append(groundtruth_path)
                break
    for track in tracks:
        for state in track.states:
            if state.timestamp > timestamp:
                break
            if state.timestamp == timestamp:
                tracks_by_timestamp[time_idx].append(track)
                break

print('Loading stonesoup metric modules...')
# %%
metrics = []
# Create Metric Generators
# ------------------------
# Here we are going to create a variety of metrics. First up is some "Basic Metrics", that simply
# computes the number of tracks, number to targets and the ratio of tracks to targets. Basic but
# useful information, that requires no additional properties.
from stonesoup.metricgenerator.basicmetrics import BasicMetrics

basic_generator = BasicMetrics()
metrics.append(basic_generator)

# %%
# Next we'll create the Optimal SubPattern Assignment (OPSA) metric generator. This metric is
# calculated at each time step, giving an overall multi-track to multi-groundtruth missed distance.
# This has two properties: :math:`p \in [1,\infty]` for outlier sensitivity and :math:`c > 1` for
# cardinality penalty. [#]_
from stonesoup.metricgenerator.ospametric import OSPAMetric
from stonesoup.measures import Euclidean

ospa_generator = OSPAMetric(c=10,
                            p=1,
                            measure=Euclidean([0, 1]))
metrics.append(ospa_generator)

# %%
# And finally we create some Single Integrated Air Picture (SIAP) metrics. Despite it's name, this
# is applicable to tracking in general and not just in relation to an air picture. This is made up
# of multiple individual metrics. [#]_
from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics

siap_generator = SIAPMetrics(position_measure=Euclidean((0, 1)),
                             velocity_measure=Euclidean((0, 1)))
metrics.append(siap_generator)

# %%
# The SIAP Metrics requires a way to associate tracks to truth, so we'll use a Track to Truth
# associator, which uses Euclidean distance measure by default.
from stonesoup.dataassociator.tracktotrack import TrackToTruth

associator = TrackToTruth(association_threshold=30)

# %%
# As a final example of a metric, we'll create a plotting metric, which is a visual way to view the
# output of our tracker.
from stonesoup.metricgenerator.plotter import TwoDPlotter

plot_generator = TwoDPlotter([0, 1], [0, 1], [0, 1])
metrics.append(plot_generator)

# %%
# Once we've created a set of metrics, these are added to a Metric Manager, along with the
# associator. The associator can be used by multiple metric generators, only being run once as this
# can be a computationally expensive process; in this case, only SIAP Metrics requires it.
from stonesoup.metricgenerator.manager import SimpleManager

metric_manager = SimpleManager(metrics,
                               associator=associator)

# %%
# Tracking and Generating Metrics
# -------------------------------
# With this basic tracker built and metrics ready, we'll now run the tracker, adding the sets of
# :class:`~.GroundTruthPath`, :class:`~.Detection` and :class:`~.Track` objects: to the metric
# manager.
print('Adding truths and tracks to metric manager...')
for timestamp in timestamps:
    time_idx = timestamps.index(timestamp)
    metric_manager.add_data(
        groundtruth_paths_by_timestamp[time_idx],
        tracks_by_timestamp[time_idx],
        overwrite=False)  # Don't overwrite, instead add above as additional data

# %%
# With the tracker run and data in the metric manager, we'll now run the generate metrics method.
# This will also generate the plot, which will be rendered automatically below, which will give a
# visual overview
plt.rcParams["figure.figsize"] = (10, 8)
print('Computing metrics...')
metrics = metric_manager.generate_metrics()

# %%
# So first we'll loop through the metrics and print out the basic metrics, which simply gives
# details on number of tracks versus targets.
for metric in metrics:
    if not any(s in metric for s in ('SIAP', 'OSPA', 'plot')):
        print(f"{metric} : {metrics.get(metric).value}")

# %%
# Next we'll take a look at the OSPA metric, plotting it to show how it varies over time. In this
# example, targets are created and remove randomly, so expect this to be fairly variable.
ospa_metric = metrics['OSPA distances']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([i.timestamp for i in ospa_metric.value], [i.value for i in ospa_metric.value])
ax.set_ylabel("OSPA distance")
ax.tick_params(labelbottom=False)
_ = ax.set_xlabel("Time")

# %%
# And finally, we'll look at the SIAP metrics, but to make these easier to visualise and understand
# we'll use a special SIAP table generator. This will colour code the results for quick visual
# indication, as well as provide a description for each metric.
from stonesoup.metricgenerator.metrictables import SIAPTableGenerator

siap_averages = {metrics.get(metric) for metric in metrics
                 if metric.startswith("SIAP") and not metric.endswith(" at times")}
siap_time_based = {metrics.get(metric) for metric in metrics if metric.endswith(' at times')}

_ = SIAPTableGenerator(siap_averages).compute_metric()

# %%
# Plotting appropriate SIAP values at each timestamp gives:

fig2, axes = plt.subplots(5)

fig2.subplots_adjust(hspace=1)

t_siaps = siap_time_based

times = metric_manager.list_timestamps()

for siap, axis in zip(t_siaps, axes):
    siap_type = siap.title[:-13]  # remove the ' at timestamp' part
    axis.set(title=siap.title, xlabel='Time', ylabel=siap_type)
    axis.tick_params(length=1)
    axis.plot(times, [t_siap.value for t_siap in siap.value])

plt.show()
print('Done.')
