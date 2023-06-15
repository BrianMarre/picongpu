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

//! @file implemets init of macro electron as co-moving with ion

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/initElectrons/CloneAdditionalAttributes.hpp"

namespace picongpu::particles::atomicPhysics2::initElectrons
{
    struct CoMoving
    {
        template<typename T_IonParticle, typename T_ElectronParticle>
        HDINLINE static void init(T_IonParticle& ion, T_ElectronParticle& electron)
        {
            CloneAdditionalAttributes::init<T_IonParticle, T_ElectronParticle>(ion, electron);

            constexpr float_X massElectronPerMassIon
                = picongpu::traits::frame::getMass<typename T_ElectronParticle::FrameType>()
                / picongpu::traits::frame::getMass<typename T_IonParticle::FrameType>();

            // init electron as co-moving with ion
            electron[momentum_] = ion[momentum_] * massElectronPerMassIon;
        }
    };
} // namespace picongpu::particles::atomicPhysics2::initElectrons
