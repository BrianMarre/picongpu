/* Copyright 2021 Brian Marre, Sergei Bastrakov
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

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            struct SetToAtomicGroundStateForChargeState
            {
                // set a given ion to its ground state for a given number of electrons
                template<typename T_Particle>
                DINLINE void operator()(T_Particle& particle, uint8_t const numberBoundElectrons)
                {
                    using Particle = T_Particle;

                    particle[boundElectrons_] = numberBoundElectrons;

                    // get current Configuration number object
                    auto configNumber = particle[atomicConfigNumber_];

                    // create blanck occupation number vector
                    auto occupationNumberVector = pmacc::math::Vector<uint8_t, configNumber.numberLevels>::create(0u);

                    uint8_t numberElectronsRemaining = numberBoundElectrons;

                    // fill from bottom up until no electrons remaining -> ground state init
                    for(uint8_t level = 1u; level <= configNumber.numberLevels; level++)
                    {
                        if(numberElectronsRemaining >= 2u * level * level)
                        {
                            (occupationNumberVector)[level - 1u] = 2u * level * level;
                            numberElectronsRemaining -= 2u * level * level;
                        }
                        else
                        {
                            (occupationNumberVector)[level - 1u] = numberElectronsRemaining;
                            break;
                        }
                    }

                    // set atomic state index
                    particle[atomicConfigNumber_] = configNumber.getAtomicStateIndex(occupationNumberVector);
                }
            };

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu