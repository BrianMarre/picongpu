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

//! @file implements Stewart-Pyatt ionization potential depression(IPD) model

#pragma once

#include "picongpu/defines.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression2::stewartPyatt
{
    namespace detail
    {
        struct StewartPyattInputStruct
        {
            //! temperature * k_Boltzman, in sim.unit.energy(), not weighted
            PMACC_ALIGN(temperatureTimesk_Boltzman, float_X);
            //! in sim.unit.length(), not weighted
            PMACC_ALIGN(debyeLength, float_X);
            //! z^Star := average(Z^2) / average(Z), with Z = (charge of ion)/e, unitless, not weighted
            PMACC_ALIGN(zStar, float_X);
        };

        struct StewartPyattAccumulationStruct
        {
            //! in weight / sim.unit.typicalNumParticlesPerMacroParticle()
            PMACC_ALIGN(sumWeightNormalizedAll, float_X);
            //! in  unitless * weight / sim.unit.typicalNumParticlesPerMacroParticle()
            PMACC_ALIGN(sumElectronWeightNormalized, float_X);

            //! in sim.unit.energy() * weight / sim.unit.typicalNumParticlesPerMacroParticle()
            PMACC_ALIGN(sumTemperatureFunctor, float_X);

            //! in  unitless * weight / sim.unit.typicalNumParticlesPerMacroParticle()
            PMACC_ALIGN(sumChargeNumber, float_X);

            //! in  unitless * weight / sim.unit.typicalNumParticlesPerMacroParticle()
            PMACC_ALIGN(sumChargeNumberSquared, float_X);
        };

    } // namespace detail

    /** implementation of Stewart-Pyatt ionization potential depression(IPD) model
     *
     * @tparam T_TemperatureFunctor term A to average over for all macro particles according to equi-partition theorem,
     * average(A) = k_B * T, must follow
     */
    template<typename T_TemperatureFunctor>
    struct StewartPyattIPDModel
    {
        using SuperCellInputStruct = detail::StewartPyattInputStruct;
        using SuperCellAccumulationStruct = detail::StewartPyattAccumulationStruct;

        /** calculate ionization potential depression
         *
         * @param input struct containing the input parameters of the Stewart-Pyatt IPD-model
         *
         * @return unit: eV, not weighted
         */
        template<uint8_t T_atomicNumber>
        HDINLINE static T_Type calculateSuperCellIPD(detail::StewartPyattInput const& input)
        {
            // eV/(sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2)
            constexpr float_X eV = sim.pic.get_eV();

            // eV/(sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2) * unitless * sim.unit.charge()^2
            //  / ( unitless * sim.unit.charge()^2 * sim.unit.time()^2 / (sim.unit.length()^3 * sim.unit.mass()))
            // = eV * sim.unit.time()^2 * sim.unit.mass()^(-1) * sim.unit.length()^(-2) * sim.unit.charge()^2 *
            // sim.unit.charge()^(-2)
            //  * sim.unit.time()^(-2) * sim.unit.length()^3 * sim.unit.mass()^1 = eV * sim.unit.length()
            // eV * sim.unit.length()
            constexpr float_X constFactor = eV * static_cast<float_X>(T_atomicNumber)
                * pmacc::math::cPow(picongpu::sim.pic.getElectronCharge(), 2u)
                / (4._X * static_cast<float_X>(picongpu::PI) * picongpu::sim.pic.getEps0());

            // (eV * sim.unit.length()) / (eV * sim.unit.length()), not weighted
            // unitless, not weighted
            float_X const K = constFactor / (input.temperatureTimesk_Boltzman * input.debyeLength);

            // eV, not weighted
            return input.temperatureTimesk_Boltzman * (math::pow(((3 * input.zStar + 1) * K + 1), 2._X / 3._X) - 1._X)
                / (2._X * (zStar + 1._X));
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression2::stewartPyatt
