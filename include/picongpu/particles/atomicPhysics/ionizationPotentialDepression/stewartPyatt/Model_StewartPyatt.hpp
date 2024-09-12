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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

//! @file implementation of ionization potential depression(IPD) calculation according of the Stewart-Pyatt Model

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/MatterIPDModel.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::tewartPyattIPD
{
    template<typename T_TemperatureFunctorTag>
    struct Model_StewartPyatt : MatterIPDModel
    {
        using TemperatureFunctor = T_TemperatureFunctorTag;

        /** calculate ionization potential depression
         *
         * @param localTemperatureEnergyBox local temperature * k_Boltzman,
         *  in sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2, not weighted
         * @param zStarBox local z^Star value, = average(q^2) / average(q), unitless, not weighted
         * @param debyeLength local debye length, sim.unit.length(), not weighted
         *
         * @return unit: eV, not weighted
         */
        template<uint8_t T_atomicNumber>
        HDINLINE static float_X calculateIPD(
            float_X const debyeLength,
            float_X const temperatureTimesk_Boltzman,
            float_X const zStar)
        {
            // eV/(sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2)
            constexpr float_X eV = static_cast<float_X>(
                picongpu::sim.unit.mass() * pmacc::math::cPow(picongpu::sim.unit.length(), 2u)
                / pmacc::math::cPow(picongpu::sim.unit.time(), 2u) * picongpu::UNITCONV_Joule_to_keV * 1e3);

            // eV/(sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2) * unitless * sim.unit.charge()^2
            //  / ( unitless * sim.unit.charge()^2 * sim.unit.time()^2 / (sim.unit.length()^3 * sim.unit.mass()))
            // = eV * sim.unit.time()^2 * sim.unit.mass()^(-1) * sim.unit.length()^(-2) * sim.unit.charge()^2 *
            // sim.unit.charge()^(-2)
            //  * sim.unit.time()^(-2) * sim.unit.length()^3 * sim.unit.mass()^1 = eV * sim.unit.length()
            // eV * sim.unit.length()
            constexpr float_X constFactor = eV * static_cast<float_X>(T_atomicNumber)
                * pmacc::math::cPow(picongpu::sim.pic.getElectronCharge(), 2u)
                / (4._X * static_cast<float_X>(picongpu::PI) * picongpu::EPS0);

            // (eV * sim.unit.length()) / (eV * sim.unit.length()) = unitless, not weighted
            // unitless, not weighted
            float_X const K = constFactor / (temperatureTimesk_Boltzman * debyeLength);

            // eV, not weighted
            return temperatureTimesk_Boltzman * (math::pow(((3 * zStar + 1) * K + 1), 2._X / 3._X) - 1._X)
                / (2._X * (zStar + 1._X));
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::tewartPyattIPD
