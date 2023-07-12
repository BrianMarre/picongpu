/* Copyright 2023 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file decelerate electrons sub-stage of atomicPhysics
 *
 * implements heating/reduction of momentum of all electron species due to accepted
 *  atomicPhysics transition
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics2/kernel/DecelerateElectrons.kernel"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::stage
{
    /** @class atomicPhysics sub-stage for a species calling the kernel per superCell
     *
     * is called once per time step for the entire local simulation volume and for
     * every isElectron species by the atomicPhysics stage by the atomicPhysicsStage
     *
     * @attention assumes DepositDeltaEnergy atomicPhysics sub-stage to have been executed previously
     *
     * @todo iterate this kernel call until all deltaEnergy is accounted for, Brian Marre, 2023
     *
     * @tparam T_ElectronSpecies species for which to call the functor
     */
    template<typename T_ElectronSpecies>
    struct DecelerateElectrons
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_ElectronSpecies
        using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ElectronSpecies>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg(MappingDesc::SuperCellSize{});

            // pointer to memory, we will only work on device, no sync required
            // init pointer to electrons and localElectronHistogramField
            auto& electrons = *dc.get<ElectronSpecies>(ElectronSpecies::FrameType::getName());
            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics2::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics2::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            using DecelerateElectrons = picongpu::particles::atomicPhysics2::kernel ::
                DecelerateElectronsKernel<T_ElectronSpecies, picongpu::atomicPhysics2::ElectronHistogram>;

            // macro for call of kernel on every superCell, see pull request #4321
            PMACC_LOCKSTEP_KERNEL(DecelerateElectrons(), workerCfg)
            (mapper.getGridDim())(
                mapper,
                electrons.getDeviceParticlesBox(),
                localElectronHistogramField.getDeviceDataBox());
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage
