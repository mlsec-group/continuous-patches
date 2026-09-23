#!/usr/bin/env python3
import pathlib
import numpy as np
from collections import defaultdict
from tqdm import trange

BASE = pathlib.Path(__file__).resolve().parent.parent / "paper_results"

# rows to include (patch_mode_prefix, temperature_or_None, label)
ROW_DEFS = [
	("velo", "warm", "Velocity, warm"),
	("velo", "cold", "Velocity, cold"),
	("timeout", "warm", "Timeout 10 Hz, warm"),
	("timeout", "cold", "Timeout 10 Hz, cold"),
	("random", None, "Random"),
	("black", None, "Black"),
]

TRAJECTORY_ORDER = ["figure8", "triangle", "u", "s"]
DISPLAY_ORDER = [60, 70, 80, 90, 100]


def walk_and_collect(base=BASE):
	"""Walk `base` for time_per_step.npy files and collect data keyed by
	(model, patch_mode, temperature, trajectory, display, pic_mode).

	Layout: model/patch_mode/[temperature/]trajectory/display/pic_mode/seed/time_per_step.npy
	DTW values are read from dtw_score.npy written next to the poses by compute_metrics.py.
	"""
	results = defaultdict(lambda: defaultdict(list))

	for fp in base.rglob("time_per_step.npy"):
		try:
			rel = fp.relative_to(base).parts
		except Exception:
			rel = fp.parts
		if len(rel) < 7:
			continue
		# model/patch_mode/[temperature/]trajectory/display/pic_mode/seed/file
		model = rel[0]
		patch_mode = rel[1]
		if rel[2] in ('warm', 'cold'):
			temperature = rel[2]
			trajectory = rel[3]
			display = rel[4]
			pic_mode = rel[5]
		else:
			temperature = None
			trajectory = rel[2]
			display = rel[3]
			pic_mode = rel[4]
		# normalize display
		display_str = str(display)
		if display_str.endswith('z'):
			display_num = display_str[:-1]
		else:
			display_num = display_str
		try:
			display_num = int(display_num)
		except Exception:
			continue
		try:
			times = np.load(fp)
			elapsed = float(np.mean(times))
		except Exception:
			continue
		entry = {'elapsed': elapsed}
		dtw_fp = fp.parent / "dtw_score.npy"
		if dtw_fp.is_file():
			try:
				entry['dtw'] = float(np.load(dtw_fp))
			except Exception:
				pass
		key = (model, patch_mode, temperature, trajectory, display_num, pic_mode)
		results[key]['values'].append(entry)
	return results


def aggregate_for_table(results):
	"""Aggregate into structure keyed by (model, patch_mode, temperature, trajectory, display) -> mean dtw, mean elapsed
	We average across pic_modes and seeds by grouping over the pic_mode dimension.
	"""
	agg = defaultdict(lambda: defaultdict(dict))
	# agg[(model,patch_mode,temperature)][(trajectory,display)] = {'dtw':mean or None, 'elapsed':mean or None}
	for (model, patch_mode, temperature, trajectory, display, pic_mode), d in results.items():
		vals = d['values']
		# collect dtw and elapsed lists
		dtws = [v['dtw'] for v in vals if 'dtw' in v]
		elaps = [v['elapsed'] for v in vals if 'elapsed' in v]
		key = (model, patch_mode, temperature)
		agg[key].setdefault((trajectory, display), {'dtw': [], 'elapsed': []})
		agg[key][(trajectory, display)]['dtw'].extend(dtws)
		agg[key][(trajectory, display)]['elapsed'].extend(elaps)

	# compute means
	for key in list(agg.keys()):
		for k2 in list(agg[key].keys()):
			dtws = agg[key][k2]['dtw']
			elaps = agg[key][k2]['elapsed']
			agg[key][k2]['dtw'] = float(np.mean(dtws)) if dtws else None
			agg[key][k2]['elapsed'] = float(np.mean(elaps)) if elaps else None
	return agg


def make_latex_table_for_model(model, agg, value_type='dtw'):
	"""Create LaTeX table string for a given model and value_type ('dtw' or 'elapsed')."""
	# header
	trajectories = TRAJECTORY_ORDER
	displays = DISPLAY_ORDER
	ncols = len(trajectories) * len(displays)
	col_spec = 'l' + 'r' * ncols
	lines = []
	lines.append('\\begin{table}[]')
	lines.append('\\centering')
	lines.append('\\begin{tabular}{' + col_spec + '}')

	# top header: trajectory groups
	header1 = ' & '.ljust(0)
	parts = ['']
	for traj in trajectories:
		parts.append('\\multicolumn{' + str(len(displays)) + '}{c}{' + traj.replace('_', ' ').title() + '}')
	lines.append(' & '.join(parts) + ' \\\\')

	# second header: display sizes
	parts = ['']
	for _ in trajectories:
		parts.extend([f'{d}\"' for d in displays])
	lines.append(' & '.join(parts) + ' \\\\')
	lines.append('\\hline')

	# rows: for each ROW_DEFS produce row of values
	for prefix, temp, label in ROW_DEFS:
		row = [label]
		# find matching keys in agg for this model
		# possible keys are (model, patch_mode, temperature)
		for traj in trajectories:
			for disp in displays:
				# matching key where patch_mode matches prefix (starts with) and temperature matches
				val = None
				# search agg keys
				for (m, patch_mode, temperature) in [k for k in agg.keys() if k[0] == model]:
					if (patch_mode.lower().startswith(prefix.lower()) or patch_mode.lower() == prefix.lower()):
						# temperature matching
						if temp is None:
							# for modes without temperature (random/black) accept temperature None
							if temperature is not None:
								continue
						else:
							if temperature != temp:
								continue
						entry = agg[(m, patch_mode, temperature)].get((traj, disp))
						if entry:
							val = entry[value_type]
							# choose first matching
							break
				if val is None:
					row.append('')
				else:
					if value_type == 'dtw':
						row.append(f"{val:.4f}")
					else:
						row.append(f"{val:.2f}")
		lines.append(' & '.join(row) + ' \\\\')

	lines.append('\\end{tabular}')
	lines.append('\\caption{' + ('DTW' if value_type == 'dtw' else 'Computation time (s)') + f' for model {model}.' + '}')
	lines.append('\\end{table}')
	return '\n'.join(lines)


def write_tables(agg):
	"""Write two .tex files per model in BASE: model_dtw.tex and model_time.tex"""
	models = sorted({k[0] for k in agg.keys()})
	for model in models:
		dtw_tex = make_latex_table_for_model(model, agg, value_type='dtw')
		time_tex = make_latex_table_for_model(model, agg, value_type='elapsed')
		(BASE / f'{model}_dtw.tex').write_text(dtw_tex)
		(BASE / f'{model}_time.tex').write_text(time_tex)

def save_raw_data(results, base=BASE):
	"""Save raw collected DTW and elapsed lists per model/patch_mode/temperature into .npz files.
	Files are named <model>__<patch_mode>__<temperature or none>.npz and contain:
	  - trajectories: array of trajectory names
	  - displays: array of display sizes
	  - dtw_lists: object array of lists (trajectory x display)
	  - elapsed_lists: object array of lists (trajectory x display)
	"""
	models = sorted({k[0] for k in results.keys()})
	for model in models:
		# find keys for this model
		model_keys = [k for k in results.keys() if k[0] == model]
		# group by (patch_mode, temperature)
		groups = {}
		for (m, patch_mode, temperature, trajectory, display, pic_mode) in model_keys:
			groups.setdefault((patch_mode, temperature), []).append((trajectory, display, pic_mode))

		for (patch_mode, temperature), items in groups.items():
			trajs = TRAJECTORY_ORDER
			displays = DISPLAY_ORDER
			T = len(trajs)
			D = len(displays)
			dtw_lists = np.empty((T, D), dtype=object)
			elapsed_lists = np.empty((T, D), dtype=object)
			for i in range(T):
				for j in range(D):
					dtw_lists[i, j] = []
					elapsed_lists[i, j] = []

			for (m, patch_mode_k, temperature_k, trajectory, display, pic_mode), d in results.items():
				if m != model:
					continue
				if patch_mode_k != patch_mode:
					continue
				# temperature may be None; require exact match
				if temperature_k != temperature:
					continue
				# map trajectory and display to indices
				try:
					ti = trajs.index(trajectory)
				except ValueError:
					continue
				try:
					di = displays.index(display)
				except ValueError:
					continue
				vals = d.get('values', [])
				dtw_vals = [v['dtw'] for v in vals if 'dtw' in v]
				elaps_vals = [v['elapsed'] for v in vals if 'elapsed' in v]
				# extend lists
				dtw_lists[ti, di].extend(dtw_vals)
				elapsed_lists[ti, di].extend(elaps_vals)

			# save
			safe_patch = patch_mode.replace('/', '_') if patch_mode else 'none'
			temp_tag = temperature if temperature is not None else 'none'
			out_fn = base / f"{model}__{safe_patch}__{temp_tag}.npz"
			np.savez_compressed(
				out_fn,
				trajectories=np.array(trajs),
				displays=np.array(displays),
				dtw_lists=dtw_lists,
				elapsed_lists=elapsed_lists,
			)

if __name__ == '__main__':
	results = walk_and_collect()
	save_raw_data(results)
	agg = aggregate_for_table(results)
	write_tables(agg)
	print('Wrote .npz and .tex files for models:', sorted({k[0] for k in agg.keys()}))