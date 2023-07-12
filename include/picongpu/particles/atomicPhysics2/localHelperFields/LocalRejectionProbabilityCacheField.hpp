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

/** @file implements a super cell local cache of of each electron histogram bin's
 *   rejectionProbability due to over subscription
 */

#pragma once

#include "picongpu/simulation_defines.hpp"
// need: picongpu::atomicPhysics2::ElectronHistogram from picongpu/param/atomicPhysics2.param

#include "picongpu/particles/atomicPhysics2/SuperCellField.hpp"
#include "picongpu/particles/atomicPhysics2/localHelperFields/RejectionProbabilityCache.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::localHelperFields
{
    /**@class superCell field of the rejectionProbabilityCache
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct LocalRejectionProbabilityCacheField
        : public SuperCellField<
              RejectionProbabilityCache<picongpu::atomicPhysics2::ElectronHistogram::numberBins>,
              T_MappingDescription,
              false /*no guards*/>
    {
        LocalRejectionProbabilityCacheField(T_MappingDescription const& mappingDesc)
            : SuperCellField<
                RejectionProbabilityCache<picongpu::atomicPhysics2::ElectronHistogram::numberBins>,
                T_MappingDescription,
                false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "LocalRejectionProbabilityCacheField";
        }
    };
} // namespace picongpu::particles::atomicPhysics2::localHelperFields
