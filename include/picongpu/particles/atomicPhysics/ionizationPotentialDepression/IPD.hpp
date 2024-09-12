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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/ModelToImplementation.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    template<typename T_IPDConfig>
    struct IPD
    {
        using FieldIPDImplementation = ModelToImplementation<T_IPDConfig::FieldIPDModel>::type;
        using MatterIPDImplementation = ModelToImplementation<T_IPDConfig::MatterIPDModel>::type;

        /** calculate all inputs for the ionization potential depression
         *
         * @tparam T_IPDIonSpeciesList list of all species partaking as ions in IPD input
         * @tparam T_IPDElectronSpeciesList list of all species partaking as electrons in IPD input
         * @tparam T_numberAtomicPhysicsIonSpecies specialization template parameter used to prevent compilation of all
         *  atomicPhysics kernels if no atomic physics species is present.
         *
         * @attention collective over all IPD species
         */
        template<
            uint32_t T_numberAtomicPhysicsIonSpecies,
            typename T_IPDIonSpeciesList,
            typename T_IPDElectronSpeciesList>
        HINLINE static void calculateIPDInput()
        {
            FieldIPDImplementation::
                calculateIPDInput<T_numberAtomicPhysicsIonSpecies, T_IPDIonSpeciesList, T_IPDElectronSpeciesList>(
                    mappingDesc);
            MatterIPDImplementation::
                calculateIPDInput<T_numberAtomicPhysicsIonSpecies, T_IPDIonSpeciesList, T_IPDElectronSpeciesList>(
                    mappingDesc);
        }

        /** check for and apply single step of pressure ionization cascade
         *
         * @attention assumes that ipd-input fields are up to date
         * @attention invalidates ipd-input fields if at least one ionization electron has been spawned
         *
         * @attention must be called once for each step in a pressure ionization cascade
         *
         * @tparam T_AtomicPhysicsIonSpeciesList list of all species partaking as ion in atomicPhysics
         * @tparam T_IPDIonSpeciesList list of all species partaking as ions in IPD input
         * @tparam T_IPDElectronSpeciesList list of all species partaking as electrons in IPD input
         *
         * @attention collective over all ion species
         */
        template<typename T_AtomicPhysicsIonSpeciesList>
        HINLINE static void applyIPDIonization(picongpu::MappingDesc const mappingDesc)
        {
            using ForEachIonSpeciesApplyPressureIonization = pmacc::meta::ForEach<
                T_AtomicPhysicsIonSpeciesList,
                s_IPD::stage::ApplyPressureIonization<boost::mpl::_1, StewartPyattIPD<T_TemperatureFunctor>>>;

            ForEachIonSpeciesApplyPressureIonization{}(mappingDesc);
        };
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
