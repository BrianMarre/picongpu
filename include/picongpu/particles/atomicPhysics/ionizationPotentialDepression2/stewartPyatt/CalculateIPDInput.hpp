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

#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/particles/algorithm/ForEach.hpp>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression2::stewartPyatt
{
    struct CalculateIPDInput
    {
        template<typename T_Worker, typename T_IonBox, typename T_AccumulationStruct>
        HDINLINE static void sumOverIons(
            T_Worker const& worker,
            T_IonBox const ionBox,
            T_AccumulationStruct& accumulationStruct)
        {
            auto forEachIonBoxEntry = pmacc::particles::algorithm::acc::makeForEach(worker, ionBox, superCellIdx);

            // end kernel if superCell already finished or if contains no particles
            if(!forEachIonBoxEntry.hasParticles())
                return;

            // init worker partial sum
            float_X sumWeight = 0._X;
            float_X sumTemperatureFunctional = 0._X;
            float_X sumChargeNumber = 0._X;
            float_X sumChargeNumberSquared = 0._X;

            forEachIonBoxEntry(
                [&sumWeight, &sumTemperatureFunctional, &sumChargeNumber, &sumChargeNumberSquared](
                    T_Worker const& worker,
                    auto& ion)
                {
                    // unitless * 1/sim.unit.typicalNumParticlesPerMacroParticle()
                    float_X const weightNormalized
                        = ion[weighting_] / picongpu::sim.unit.typicalNumParticlesPerMacroParticle();

                    // sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2 * weight /
                    // sim.unit.typicalNumParticlesPerMacroParticle()
                    sumTemperatureFunctional
                        += T_TemperatureFunctor::term(ion, precisionCast<float_64>(weightNormalized));

                    // weight / sim.unit.typicalNumParticlesPerMacroParticle()
                    sumWeight += weightNormalized;

                    // sim.unit.charge()
                    constexpr auto elementaryCharge = -picongpu::sim.pic.getElectronCharge();

                    // sim.unit.charge() / sim.unit.charge() = unitless
                    auto const chargeNumber = picongpu::traits::attribute::getCharge(1._X, ion) / elementaryCharge;

                    // unitless * weight / sim.unit.typicalNumParticlesPerMacroParticle()
                    sumChargeNumber += weightNormalized * chargeNumber;
                    // unitless * weight / sim.unit.typicalNumParticlesPerMacroParticle()
                    sumChargeNumberSquared += weightNormalized * pmacc::math::cPow(chargeNumber, 2u);
                });

            // use atomic add to write worker partial sums for species back collective variable
            alpaka::atomicAdd(
                worker.getAcc(),
                &(accumulationStruct.sumWeightNormalizedAll),
                sumWeight,
                ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(
                worker.getAcc(),
                &(accumulationStruct.sumTemperatureFunctor()),
                sumTemperatureFunctional,
                ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(
                worker.getAcc(),
                &(accumulationStruct.sumChargeNumber()),
                sumChargeNumber,
                ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(
                worker.getAcc(),
                &(accumulationStruct.sumChargeNumberSquared()),
                sumChargeNumberSquared,
                ::alpaka::hierarchy::Threads{});
        }

        template<typename T_Worker, typename T_ElectronBox, typename T_AccumulationStruct>
        HDINLINE static void sumOverElectrons(
            T_Worker const& worker,
            T_ElectronBox const electronBox,
            T_AccumulationStruct& accumulationStruct)
        {
            auto forEachElectronBoxEntry
                = pmacc::particles::algorithm::acc::makeForEach(worker, electronBox, superCellIdx);

            // end kernel if superCell contains no particles
            if(!forEachElectronBoxEntry.hasParticles())
                return;

            // init worker partial sum
            float_X sumWeight = 0._X;
            float_X sumTemperatureFunctional = 0._X;
            float_X sumElectronWeight = 0._X;

            forEachElectronBoxEntry(
                [&sumWeight, &sumTemperatureFunctional, &sumElectronWeight](T_Worker const& worker, auto& electron)
                {
                    // unitless * 1/sim.unit.typicalNumParticlesPerMacroParticle()
                    float_X const weightNormalized
                        = electron[weighting_] / picongpu::sim.unit.typicalNumParticlesPerMacroParticle();

                    // sim.unit.mass() * sim.unit.length()^2 / sim.unit.time()^2 * weight /
                    // sim.unit.typicalNumParticlesPerMacroParticle()
                    sumTemperatureFunctional
                        += T_TemperatureFunctor::term(electron, precisionCast<float_64>(weightNormalized));

                    // weight / sim.unit.typicalNumParticlesPerMacroParticle()
                    sumWeight += weightNormalized;
                    sumElectronWeight += weightNormalized;
                });

            // write worker partial sums for species to accumulationStruct
            alpaka::atomicAdd(
                worker.getAcc(),
                &(accumulationStruct.sumWeightNormalizedAll),
                sumWeight,
                ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(
                worker.getAcc(),
                &(accumulationStruct.sumTemperatureFunctor),
                sumTemperatureFunctional,
                ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(
                worker.getAcc(),
                &(accumulationStruct.sumElectronWeightNormalized),
                sumElectronWeight,
                ::alpaka::hierarchy::Threads{});
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression2::stewartPyatt
