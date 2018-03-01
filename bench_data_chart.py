import numpy as np
from dateutil.parser import parse
import datetime as dt


domain = ["10:00:00", "21:00:00"]
global_px = 732 #pixels
bench_data = [["10:05:55", "10:09:00"], ["10:45:39", "10:49:10"], ["13:08:02", "13:08:11"], ["13:08:19", "13:08:27"], ["13:08:36", "13:09:44"], ["13:46:20", "13:50:33"], ["14:16:30", "14:16:55"], ["14:17:52", "14:18:36"], ["14:19:28", "14:20:16"], ["14:22:45", "14:23:37"], ["16:38:02", "16:39:49"], ["19:08:39", "19:08:56"], ["19:13:39", "19:17:57"]]

durations = []
ticks = []
pxs = []

def get_interval_seconds(start, stop):
	return (stop.hour*3600 + stop.minute*60 + stop.second) - (start.hour*3600 + start.minute*60 + start.second)

def start_elapsed(start):
	global_start = parse(domain[0])
	return (start.hour*3600 + start.minute*60 + start.second) - (global_start.hour*3600 + global_start.minute*60 + global_start.second)

def calc_pixels(tick):
	global_domain =  start_elapsed(parse(domain[1]))
	return (float(tick)/global_domain)*global_px

for i in bench_data:
	start, stop = parse(i[0]), parse(i[1])  
	duration = get_interval_seconds(start, stop)
	tick = start_elapsed(start)
	px = calc_pixels(tick)
	#duration = parse(i[1] - i[0])
	durations.append(duration)
	ticks.append(tick)
	pxs.append(int(px))


print durations
print pxs