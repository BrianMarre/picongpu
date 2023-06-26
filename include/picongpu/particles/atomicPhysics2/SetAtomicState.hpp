/* Copyright 2023 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software you can redistribute it andor modify
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

//! @file implements functor for setting to atomicState

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/traits/GetAtomicConfigNumber"

#include <pmacc/static_assert.hpp>

namespace picongpu::particles::atomicPhysics2
{
    struct SetAtomicState
    {
        template<typename T_Ion, typename T_ConfigNumber>
        DINLINE void operator()(T_Ion& ion, typename T_ConfigNumber::Idx newAtomicConfigNumber)
        {
            ion[atomicConfigNumber_] = newAtomicConfigNumber;
            ion[boundElectrons_] = T_ConfigNumber::getBoundElectrons(newConfigNumber);

            PMACC_DEVICE_ASSERT_MSG(
                T_ConfigNumber::getBoundElectrons(newConfigNumber) <= T_ConfigNumber::atomicNumber,
                "Number of bound electrons must be <= Z");
        }
    };

} // namespace picongpu::particles::atomicPhysics2
