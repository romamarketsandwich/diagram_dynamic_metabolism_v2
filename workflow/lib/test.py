import json
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynamic_catalysis_model import (
	build_rate_fn_kwargs,
	calculate_tree_weights,
	contract_node,
	cycle_flux_fn,
	cycle_coefficient,
	cycle_inflows,
	generate_spanning_trees,
	is_valid_root,
	load_simulation_settings,
	make_rate_fn,
	run_dycat_model,
)


def _two_state_graph():
	graph = nx.Graph()
	graph.add_edge(0, 1)
	return graph


def _constant_weights(**params):
	return {(0, 1): 2.0, (1, 0): 3.0}


def _constant_rate(**params):
	return 1.0


def _constant_midpoint(**params):
	return 0.0


def _rate_settings():
	return {
		"cycles_1": [[0, 1]],
		"cycles_2": [[1, 0]],
		"absorption_edge_list": None,
		"max_ss_rate_fn": _constant_rate,
		"weight_mapping_fn": _constant_weights,
		"osc_midpoint_fn": _constant_midpoint,
		"graph_fn": _two_state_graph,
	}


def test_cycle_flux_fn_calculates_normalized_turnover_and_futile_flux():
	turnover_flux, futile_flux = cycle_flux_fn(
		[[0, 1]],
		[[1, 0]],
		_two_state_graph(),
		max_ss_rate_fn=_constant_rate,
		weight_mapping_fn=_constant_weights,
	)

	assert turnover_flux == pytest.approx(6 / 5)
	assert futile_flux == pytest.approx(1.0)


def test_rate_fn_integrates_parameter_mapping_and_four_parameter_branch():
	rate_fn = make_rate_fn(build_rate_fn_kwargs(_rate_settings()))

	turnover_flux, futile_flux = rate_fn([10.0, 20.0, -1.0, 1.0])

	assert turnover_flux == pytest.approx(6 / 5)
	assert futile_flux == pytest.approx(1.0)


def test_rate_fn_integrates_three_parameter_amplitude_branch():
	rate_fn = make_rate_fn(build_rate_fn_kwargs(_rate_settings()))

	turnover_flux, futile_flux = rate_fn([10.0, 20.0, 10.0])

	assert turnover_flux == pytest.approx(6 / 5)
	assert futile_flux == pytest.approx(1.0)


def test_rate_fn_rejects_parameter_vectors_with_wrong_length():
	rate_fn = make_rate_fn(build_rate_fn_kwargs(_rate_settings()))

	with pytest.raises(ValueError, match="length 3 or 4"):
		rate_fn([10.0, 20.0])


def test_cycle_coefficient_rejects_cycles_with_fewer_than_two_nodes():
	with pytest.raises(AssertionError):
		cycle_coefficient([0], {(0, 0): 1.0})


def test_settings_file_round_trip_builds_expected_rate_function_kwargs(tmp_path):
	settings = {
		"cycles_1": [[0, 1]],
		"alpha_32": 2.5,
		"k_bT": 1.0,
	}
	settings_path = tmp_path / "simulation.json"
	settings_path.write_text(json.dumps(settings))

	loaded_settings = load_simulation_settings(settings_path)
	rate_kwargs = build_rate_fn_kwargs(loaded_settings)

	assert rate_kwargs["cycles_1"] == [[0, 1]]
	assert rate_kwargs["alpha_32"] == 2.5
	assert rate_kwargs["k_bT"] == 1.0
	assert rate_kwargs["cycles_2"] is None


def test_calculate_tree_weights_handles_roots_and_absorption_edges():
	graph = nx.path_graph(3)
	weights = {
		(1, 0): 2.0,
		(2, 1): 3.0,
		(1, 2): 5.0,
		(0, 1): 7.0,
	}

	assert calculate_tree_weights(graph, weights, root_nodes=[0]) == pytest.approx(6.0)
	assert calculate_tree_weights(
		graph, weights, absorption_edge_list=[(1, 0)]
	) == pytest.approx(6.0)


def test_contract_node_remaps_weights_and_absorption_edges():
	graph = nx.Graph([(0, 1), (1, 2), (0, 2)])
	weights = {
		(0, 1): 1.0,
		(1, 0): 2.0,
		(0, 2): 3.0,
		(2, 0): 4.0,
		(1, 2): 5.0,
		(2, 1): 6.0,
	}

	new_weights, new_graph, new_absorption_edges = contract_node(
		0, 1, graph, weights, absorption_edge_list=[(1, 2), (2, 1)]
	)

	assert set(new_graph) == {0, 2}
	assert new_weights[(0, 2)] == pytest.approx(8.0)
	assert new_weights[(2, 0)] == pytest.approx(10.0)
	assert new_absorption_edges == [(0, 2), (2, 0)]


def test_root_validation_and_spanning_tree_generation_respect_absorption():
	graph = nx.path_graph(3)
	absorption_edges = [(1, 0)]

	assert is_valid_root(graph, 0, absorption_edges)
	assert not is_valid_root(graph, 1, absorption_edges)
	assert is_valid_root(graph, 2, [])

	spanning_trees = generate_spanning_trees(graph, absorption_edges)

	assert len(spanning_trees) == 1
	assert set(spanning_trees[0].edges) == {(0, 1), (1, 2)}

	assert generate_spanning_trees(
		nx.path_graph(2), [(0, 1), (1, 0)]
	) == []


def test_cycle_helpers_calculate_coefficient_and_inflow():
	graph = nx.Graph([(0, 1), (1, 2), (0, 2)])
	weights = {
		(0, 1): 2.0,
		(1, 2): 3.0,
		(2, 0): 5.0,
		(1, 0): 7.0,
		(2, 1): 11.0,
		(0, 2): 13.0,
	}

	assert cycle_coefficient([0, 1, 2], weights) == pytest.approx(30.0)
	assert cycle_inflows([0, 1, 2], graph, weights) == pytest.approx(1.0)
	external_graph = nx.Graph([(0, 1), (1, 2)])
	external_weights = {
		(0, 1): 2.0,
		(1, 0): 7.0,
		(0, 2): 13.0,
		(2, 0): 5.0,
	}
	assert cycle_inflows([0, 1], external_graph, external_weights) > 0
	with pytest.raises(AssertionError):
		cycle_inflows([0], graph, weights)


def test_run_dycat_model_executes_workflow_and_writes_outputs(tmp_path, monkeypatch):
	class FakeBoundaryExplorer:
		calls = 0

		def __init__(self, **settings):
			FakeBoundaryExplorer.calls += 1
			self.mat = np.array([[1.0]])
			if FakeBoundaryExplorer.calls == 1:
				self.mat_pars = np.array([[[10.0, 20.0, -1.0, 1.0]]])
			else:
				self.mat_pars = np.array([[[10.0, 20.0, 10.0]]])

		def get_initial_points(self, n_points):
			assert n_points == 10

		def extend_boundary(self, **args):
				assert args["folder_save"].startswith(str(tmp_path))

	def fake_import_module(name):
		assert name == "utils"
		return SimpleNamespace(
			constant_weights=_constant_weights,
			two_state_graph=_two_state_graph,
			constant_rate=_constant_rate,
			constant_midpoint=_constant_midpoint,
		)

	original_make_rate_fn = make_rate_fn

	def instrumented_make_rate_fn(hyperparams):
		if hyperparams["max_ss_rate_fn"].__name__ == "dummy_max_rate_fn":
			assert hyperparams["osc_midpoint_fn"]() == 0
		return original_make_rate_fn(hyperparams)

	monkeypatch.setattr("dynamic_catalysis_model.importlib.import_module", fake_import_module)
	monkeypatch.setattr("dynamic_catalysis_model.BF.BoundaryExplorer", FakeBoundaryExplorer)
	monkeypatch.setattr("dynamic_catalysis_model.make_rate_fn", instrumented_make_rate_fn)

	settings = {
		"folder_save": str(tmp_path / "results"),
		"name_save": "integration",
		"cycles_1": [[0, 1]],
		"cycles_2": [[1, 0]],
		"absorption_edge_list": None,
		"max_ss_rate_fn": "constant_rate",
		"weight_mapping_fn": "constant_weights",
		"osc_midpoint_fn": "constant_midpoint",
		"graph_fn": "two_state_graph",
		"col_ar_min": 0.0,
		"col_ar_max": 1.0,
		"row_ar_min": 0.0,
		"row_ar_max": 1.0,
		"n_grid": 1,
		"pars_limit_min": 0.0,
		"pars_limit_max": 1.0,
		"npars": 4,
		"extr_uniform": False,
		"L_project": False,
		"plotting": False,
		"verbose": False,
		"dofirstmutate": False,
		"dopullcentroids": False,
		"dopulltangents": False,
	}
	config_path = tmp_path / "config.json"
	config_path.write_text(json.dumps(settings))

	output_folder = Path(run_dycat_model(config_path))

	assert output_folder.is_dir()
	assert np.array_equal(
		np.load(output_folder / "mat_integration_last.npy"),
		np.array([[1.0]])
	)
	assert np.array_equal(
		np.load(output_folder / "mat_pars_integration_last.npy"),
		np.array([[[10.0, 20.0, -1.0, 1.0]]])
	)
	assert np.allclose(
		np.load(output_folder / "sub_cycle_fluxes_integration_last.npy"),
		np.array([[1.2], [1.2]])
	)
	assert json.loads((output_folder / "integration_config.json").read_text()) == settings

	settings["folder_save"] = str(tmp_path / "auto_results")
	settings["max_ss_rate_fn"] = None
	settings["osc_midpoint_fn"] = None
	auto_config_path = tmp_path / "auto_config.json"
	auto_config_path.write_text(json.dumps(settings))

	auto_output_folder = Path(run_dycat_model(auto_config_path))

	assert auto_output_folder.is_dir()
	assert (auto_output_folder / "mat_integration_last.npy").is_file()
	assert (auto_output_folder / "sub_cycle_fluxes_integration_last.npy").is_file()
