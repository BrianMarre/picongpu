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

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/ionizationPotentialDepression/stage/CalculateIPDInput.hpp"
#include "picongpu/particles/atomicPhysics2/ionizationPotentialDepression/stage/FillIPDSumFields_Electron.hpp"
#include "picongpu/particles/atomicPhysics2/ionizationPotentialDepression/stage/FillIPDSumFields_Ion.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
{
    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics2::ionizationPotentialDepression;

    /** implementation of Stewart-Pyatt ionization potential depression(IPD) model
     *
     * @tparam T_TemperatureFunctor term A to average over for all macro particles according to equi-partition theorem,
     * average(A) = k_B * T, must follow
     */
    template<typename T_TemperatureFunctor>
    struct StewartPyattIPD
    {
        //! reset IPD support infrastructure before we accumulate over particles to calculate new IPD Inputs
        HINLINE static void reset()
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localSumWeightAllField = *dc.get<s_IPD::localHelperFields::SumWeightAllField>("SumWeightAllField");
            auto& localSumTemperatureFunctionalField
                = *dc.get<s_IPD::localHelperFields::SumTemperatureFunctionalField>("SumTemperatureFunctionalField");

            auto& localSumWeightElectronField
                = *dc.get<s_IPD::localHelperFields::SumWeigthElectronsField>("SumWeightElectronsField");

            auto& localSumChargeNumberIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberIonsField>("SumChargeNumberIonsField");
            auto& localSumChargeNumberSquaredIonsField
                = *dc.get<s_IPD::localHelperFields::SumChargeNumberSquaredIonsField>(
                    "SumChargeNumberSquaredIonsField");

            localSumWeightAllField.getDeviceBuffer().setValue(0._X);
            localSumTemperatureFunctionalField.getDeviceBuffer().setValue(0._X);
            localSumWeightElectronField.getDeviceBuffer().setValue(0._X);
            localSumChargeNumberIonsField.getDeviceBuffer().setValue(0._X);
            localSumChargeNumberSquaredIonsField.getDeviceBuffer().setValue(0._X);
        }

        //! do all precalculation work for IPD
        HINLINE static void calculateIPDInput(picongpu::MappingDesc const mappingDesc)
        {
            // figure out species to reduce
            //{
            //! list of all atomicPhysics partaking electron species
            using AtomicPhysicsElectronSpecies =
                typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsElectron<>>::type;
            //! list of all only IPD partaking electron species
            using OnlyIPDElectronSpecies =
                typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isOnlyIPDElectron<>>::type;

            //! list of all atomicPhysics partaking ion species
            using AtomicPhysicsElectronSpecies =
                typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsIon<>>::type;
            //! list of all only IPD partaking ion species
            using OnlyIPDIonSpecies =
                typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isOnlyIPDIon<>>::type;

            //! list of all electron species for IPD
            using IPDElectronSpecies = MakeSeq_t<AtomicPhysicsElectronSpecies, OnlyIPDElectronSpecies>;
            //! list of all ion species for IPD
            using IPDIonSpecies = MakeSeq_t<AtomicPhysicsIonSpecies, OnlyIPDIonSpecies>;
            //}

            using ForEachElectronSpeciesFillSumFields = pmacc::meta::
                ForEach<IPDIonSpecies, s_IPD::stage::FillIPDSumFields_Electron<boost::mpl::_1, T_TemperatureFunctor>>;
            using ForEachIonSpeciesFillSumFields = pmacc::meta::
                ForEach<IPDIonSpecies, s_IPD::stage::FillIPDSumFields_Ion<boost::mpl::_1, T_TemperatureFunctor>>;

            using ForEachIonSpeciesFillSumFields = pmacc::meta::
                ForEach<IPDIonSpecies, s_IPD::stage::FillIPDSumFields_Ion<boost::mpl::_1, T_TemperatureFunctor>>;

            // reset IPD SumFields
            reset();

            ForEachElectronSpeciesFillSumFields{}(mappingDesc);
            ForEachIonSpeciesFillSumFields{}(mappingDesc);

            s_IPD::stage::CalculateIPDInput()(mappingDesc);
        }

        /** calculate ionization potential depression
         *
         * @tparam T_atomicNumber atomicNumber of ion species
         *
         * @param temperatureTimesk_Boltzman k_B * T in UNIT_MASS * UNIT_LENGTH^2 / UNIT_TIME^2
         * @param zStar = average(q^2) / average(q) ; q = charge number of ions, unitless
         * @param debyeLength in UNIT_LENGTH
         *
         * @todo switch to per species pre-calculated IPD field, Brian Marre 2024
         *
         * @return unit UNIT_MASS * UNIT_LENGTH^2 / UNIT_TIME^2
         */
        template<uint8_t T_atomicNumber>
        HDINLINE static float_X getIPD(
            float_X const temperatureTimesk_Boltzman,
            float_X const zStar,
            float_X const debyeLength)
        {
            // unitless * UNIT_CHARGE^2 / ( unitless * UNIT_CHARGE^2 * UNIT_TIME^2 / (UNIT_LENGTH^3 * UNIT_MASS))
            // UNIT_MASS * UNIT_LENGTH^3 / UNIT_TIME^2
            constexpr float_X scaleFactor = static_cast<float_X>(T_atomicNumber)
                * pmacc::math::cPow(picongpu::ELECTRON_CHARGE, 2u)
                / (4._X * static_cast<float_X>(picongpu::PI) * picongpu::EPS0);

            // (UNIT_MASS * UNIT_LENGTH^3 / UNIT_TIME^2) / (UNIT_MASS * UNIT_LENGTH^2 / UNIT_TIME^2 * UNIT_LENGTH)
            // unitless
            float_X const K = scaleFactor / (temperatureTimesk_Boltzman * debyeLength);

            // UNIT_MASS * UNIT_LENGTH^2 / UNIT_TIME^2
            return temperatureTimesk_Boltzman * (math::pow(((3 * zStar + 1) * K + 1), 2._X / 3._X) - 1._X)
                / (2._X * (zStar + 1));
        }
    };
} // namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
