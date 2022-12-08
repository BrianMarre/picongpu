/* Copyright 2022 Brian Marre
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

/** @file implements the local timeStepField for each superCell
 *
 * timeStep length for the current atomicPhysics iteration in each superCell
 */

#pragma once

#include "picongpu/particles/atomicPhysics2/SuperCellField.hpp"

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics2
{
    /**@class superCell field of the current timeStep:float_X for one atomicPhysics iteration
     *
     */
    template<typename T_MappingDescription>
    struct LocalTimeStepField : public SuperCellField<float_X, T_MappingDescription>
    {
        LocalTimeStepField(T_MappingDescription const& mappingDesc)
            : SuperCellField<float_X, T_MappingDescription>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "LocalTimeStepField";
        }
    };
} // namespace picongpu::particles::atomicPhysics2
