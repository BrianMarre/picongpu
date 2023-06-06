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

//! @file implements cloning all particle attributes, excluding momentum, particleID and multiMask, from ion to electron

#pragma once

#include "picongpu/simulation_defines.hpp"

namespace picongpu::particles::atomicPhysics2::initElectrons
{
    struct CloneAdditionalAttributes
    {
        /** clone all add attributes that exist in both electron and ion species
         *
         * excludes:
         *  - particleId, new particle --> new ID required, init by default
         *  - multiMask, faster to set hard than copy, set in Kernel directly
         *  - momentum, is mass dependent and therefore always changes
        */
        template<typename T_IonParticle, typename T_ElectronParticle>
        HDINLINE static void init(
            T_IonParticle& ion,
            // cannot be const even though we do not write to the ion
            T_ElectronParticle& electron)
        {
            namespace partOp = pmacc::particles::operations;

            auto targetElectronClone = partOp::deselect<pmacc::mp_list<multiMask, momentum>>(electron);

            // otherwise this deselect will create incomplete type compile error
            partOp::assign(targetElectronClone, partOp::deselect<particleId>(ion));
        }
    };
} // namespace picongpu::particles::atomicPhysics2::initElectrons
