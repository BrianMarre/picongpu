/* Copyright 2024 Brian Marre
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

//! @file no ionization potential depression implementation

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/ionizationPotentialDepression/IPDInterface.hpp"

namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
{
    struct NoIPD : IPDInterface
    {
        //! create all HelperFields required by the IPD model
        HINLINE static void createHelperFields()
        {
        }

        HINLINE static void calculateIPDInput(picongpu::MappingDesc const mappingDesc)
        {
        }

        /** calculate ionization potential depression
         *
         * @returns 0._X
         */
        HDINLINE static float_X calculateIPD()
        {
            return 0._X;
        }
    }
} // namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
