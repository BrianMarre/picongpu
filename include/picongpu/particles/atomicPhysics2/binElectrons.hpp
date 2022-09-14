/* Copyright 2022 Sergei Bastrakov, Brian Marre, Axel Huebl, Rene Widera
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

#pragma once

#include <cstdint>
#include <cstring>

#inlcude "picongpu/simulation_defines.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>
#include <pmacc/type/Area.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>

#include "picongpu/particles/atomicPhyiscs2/binElectrons.kernel"

/** @file implements kernel call 
 *
 */

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhyiscs2
        {
            /** atomicPhysics sub-stage calling binElectronKernel(binLectrons.kernel) once for every local superCell
             *
             * is called once per time step for the entire local simulation volume and for
             * every isElectron species by the atomicPhysics stage
             */
            template<typename T_ElectronSpecies>
            struct binElectrons
            {
                // might be alias, from here on out no more
                using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_ElectronSpecies>;
                using LocalHistogramField = typename particles::atomicPhyiscs2::electronDistribution::LocalHistogramField;

                static void operator()(picongpu::MappingDesc const mappingDesc) const
                {

                    // full local domain, no guards
                    pmacc::AreaMapping< CORE + BORDER, MappingDesc> mapper(mappingDesc);
                    pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

                    constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
                        pmacc::math::CT::volume<MappingDesc::SuperCellSize>::type::value>::value;

                    // pointer to memory, we will only work on device, no sync required
                    // init pointer to electrons and localElectronHistogramField
                    auto& electrons = *dc.get<ElectronSpecies>(ElectronSpecies::FrameType::getName(), true);
                    auto& localElectronHistogramField = *dc.get<
                        LocalHistogramField<
                            atomicPhyiscs2::ElectronHistogram,
                            picongpu::MappingDesc>
                        >("Electron_localHistogramField", true);

                    // macro for call of kernel
                    PMACC_KERNEL( binElectronsKernel<
                        ElectronSpecies,
                        atomicPhyiscs2::ElectronHistogram>,
                        numWorkers) // kernel to call
                    (mapper.getGridDim(), // how many blocks(superCells in local domain)
                     numWorkers // how many threads per block
                     )(mapper, electrons.getDeviceParticlesBox(),
                       (localElectronHistogramField->localHistogramField)->getDeviceBuffer());
                }
            }
        } // namespace atomicPhyiscs2
    } // namespace particles
} // namespace picongpu