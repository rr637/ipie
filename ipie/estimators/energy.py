# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Fionn Malone <fmalone@google.com>
#

from typing import Union
import numpy as np
import wandb
import plum

from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.local_energy_batch import (
    local_energy_batch,
    local_energy_multi_det_trial_batch,
)
from ipie.estimators.local_energy_noci import local_energy_noci
from ipie.estimators.local_energy_sd import (
    local_energy_single_det_uhf_batch,
    local_energy_single_det_ghf_batch,
)
from ipie.estimators.local_energy_sd_isdf import local_energy_single_det_isdf_batch_gpu
from ipie.estimators.local_energy_sd_chunked import (
    local_energy_single_det_uhf_batch_isdf_chunked_gpu,
)
from ipie.estimators.local_energy_wicks import (
    local_energy_multi_det_trial_wicks_batch,
    local_energy_multi_det_trial_wicks_batch_opt,
    local_energy_multi_det_trial_wicks_batch_opt_chunked,
)
from ipie.estimators.local_energy_kpt_sd import local_energy_kpt_single_det_uhf
from ipie.estimators.local_energy_kpt_sd_isdf import local_energy_kpt_single_det_uhf_isdf_gpu
from ipie.estimators.local_energy_kpt_sd_chunked import local_energy_kpt_single_det_uhf_chunked
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.hamiltonians.isdf import GenericRealISDF
from ipie.hamiltonians.generic_chunked import GenericRealCholChunked
from ipie.hamiltonians.chunked_isdf import GenericRealISDFChunked
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.particle_hole import (
    ParticleHole,
    ParticleHoleNaive,
    ParticleHoleNonChunked,
    ParticleHoleSlow,
)
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm
from ipie.hamiltonians.kpt_isdf_hamiltonian import KptISDF
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.backend import arraylib as xp
from ipie.walkers.ghf_walkers import GHFWalkers


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: Union[GenericRealChol, GenericRealCholChunked],
    walkers: UHFWalkers,
    trial: SingleDet,
):
    return local_energy_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericComplexChol,
    walkers: UHFWalkers,
    trial: SingleDet,
):
    return local_energy_single_det_uhf_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHoleNaive,
):
    return local_energy_multi_det_trial_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHole,
):
    return local_energy_multi_det_trial_wicks_batch_opt_chunked(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHoleNonChunked,
):
    return local_energy_multi_det_trial_wicks_batch_opt(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHoleSlow,
):
    return local_energy_multi_det_trial_wicks_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(system: Generic, hamiltonian: GenericRealChol, walkers: UHFWalkers, trial: NOCI):
    return local_energy_noci(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: KptComplexChol, walkers: UHFWalkers, trial: KptSingleDet
):
    return local_energy_kpt_single_det_uhf(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: KptComplexCholSymm, walkers: UHFWalkers, trial: KptSingleDet
):
    return local_energy_kpt_single_det_uhf(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: KptComplexCholChunked, walkers: UHFWalkers, trial: KptSingleDet
):
    return local_energy_kpt_single_det_uhf_chunked(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(system: Generic, hamiltonian: KptISDF, walkers: UHFWalkers, trial: KptSingleDet):
    return local_energy_kpt_single_det_uhf_isdf_gpu(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: GenericRealChol, walkers: GHFWalkers, trial: SingleDetGHF
):
    return local_energy_single_det_ghf_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: GenericComplexChol, walkers: GHFWalkers, trial: SingleDetGHF
):
    return local_energy_single_det_ghf_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: GenericRealISDF, walkers: UHFWalkers, trial: SingleDet
):
    return local_energy_single_det_isdf_batch_gpu(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: GenericRealISDFChunked, walkers: UHFWalkers, trial: SingleDet
):
    return local_energy_single_det_uhf_batch_isdf_chunked_gpu(system, hamiltonian, walkers, trial)


class EnergyEstimator(EstimatorBase):
    def __init__(
        self,
        system=None,
        ham=None,
        trial=None,
        filename=None,
    ):
        super().__init__()
        self._eshift = 0.0
        self.scalar_estimator = True
        self._data = {
            "ENumer": 0.0j,
            "EDenom": 0.0j,
            "ETotal": 0.0j,
            "E1Body": 0.0j,
            "E2Body": 0.0j,
            "ENumer_WalkerVar": 0.0j,
            "E1Body_WalkerVar": 0.0j,
            "E2Body_WalkerVar": 0.0j,
            "ENumer_WalkerSTD": 0.0j,
            "E1Body_WalkerSTD": 0.0j,
            "E2Body_WalkerSTD": 0.0j,
            "ETotal_WalkerVar": 0.0j,
            "ETotal_WalkerSTD": 0.0j,
        }
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True
        self.ascii_filename = filename

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        trial.calc_greens_function(walkers)
        # Need to be able to dispatch here
        energy = local_energy(system, hamiltonian, walkers, trial)  # [N_walkers, 3]
        this_E_var = np.var(energy.real, axis=0)  # take variance along num_walkers
        this_E_std = np.std(energy.real, axis=0)

        # stats wrt walkers
        self._data["ENumer_WalkerVar"] = float(this_E_var[0])
        self._data["E1Body_WalkerVar"] = float(this_E_var[1])
        self._data["E2Body_WalkerVar"] = float(this_E_var[2])
        self._data["ENumer_WalkerSTD"] = float(this_E_std[0])
        self._data["E1Body_WalkerSTD"] = float(this_E_std[1])
        self._data["E2Body_WalkerSTD"] = float(this_E_std[2])

        E_numer = self._data["ENumer"]            # = sum_i w_i E_i
        w_sum   = np.sum(walkers.weight)         # = sum_i w_i

        # Build the samples for X and Y:
        X = walkers.weight * energy[:,0].real     # shape (N,)
        Y = walkers.weight                        # shape (N,)

        # sample means (just to be explicit):
        X_mean = E_numer
        Y_mean = w_sum

        # sample variances & covariance (unbiased, ddof=1):
        var_X   = np.var(X, ddof=1)
        var_Y   = np.var(Y, ddof=1)
        cov_XY  = np.cov(X, Y, ddof=1)[0,1]

        # atio variance:
        var_etot = ( var_X / Y_mean**2
                + X_mean**2 * var_Y / Y_mean**4
                - 2* X_mean * cov_XY / Y_mean**3 )
        std_etot = np.sqrt(var_etot)

        self._data["ETotal_WalkerVar"] = float(np.real(var_etot))
        self._data["ETotal_WalkerSTD"] = float(np.real(std_etot))

        self._data["ENumer"] = xp.sum(walkers.weight * energy[:, 0].real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        self._data["E1Body"] = xp.sum(walkers.weight * energy[:, 1].real)
        self._data["E2Body"] = xp.sum(walkers.weight * energy[:, 2].real)

        return self.data

    def get_index(self, name):
        index = self._data_index.get(name, None)
        if index is None:
            raise RuntimeError(f"Unknown estimator {name}")
        return index

    def post_reduce_hook(self, data):
        # get indices from reverse mapping list
        ix_total = self._data_index["ETotal"]
        ix_nume = self._data_index["ENumer"]
        ix_deno = self._data_index["EDenom"]

        # normalize over total walker weight
        data[ix_total] = data[ix_nume] / data[ix_deno]  # norm ETotal
        ix_E1_nume = self._data_index["E1Body"]
        data[ix_E1_nume] = data[ix_E1_nume] / data[ix_deno]  # norm E1Body
        ix_E2_nume = self._data_index["E2Body"]
        data[ix_E2_nume] = data[ix_E2_nume] / data[ix_deno]  # norm E2Body