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

//! @file reset deltaWeight entry of all histogram bin

#pragma once

// need picongpu::atomicPhysics2::ElectronHistogram from atomicPhysics2.param
#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics2/kernel/ResetDeltaWeightElectronHistogram.kernel"

#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/type/Area.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::stage
{
    /** ResetDeltaWeightElectronHistogram atomic physics sub-stage
     *
     * reset deltaWeight entry for each electron histogram bin to 0
     */
    struct ResetDeltaWeightElectronHistogram
    {
        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg(MappingDesc::SuperCellSize{});

            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics2::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics2::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            // macro for call of kernel for every superCell, see pull request #4321
            PMACC_LOCKSTEP_KERNEL(
                picongpu::particles::atomicPhysics2::kernel::ResetDeltaWeightElectronHistogramKernel<
                    picongpu::atomicPhysics2::ElectronHistogram>(),
                workerCfg)
            (mapper.getGridDim())(mapper, localElectronHistogramField.getDeviceDataBox());

            /// @todo implement photon histogram, Brian Marre, 2023
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage
