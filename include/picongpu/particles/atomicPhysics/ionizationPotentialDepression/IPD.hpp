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

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/ApplyIPDIonization.hpp"
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

        /** append ipd input to kernelInput and do a PMACC_LOCKSTEP_KERNEL call for T_kernel
         *
         * @tparam T_Kernel kernel to call
         * @param kernelInput stuff to pass to the kernel, before the ionization potential depression input
         */
        template<typename T_Kernel, uint32_t chunkSize, typename... T_KernelInput>
        HINLINE static void callKernelWithIPDInput(
            pmacc::DataConnector& dc,
            pmacc::AreaMapping<CORE + BORDER, picongpu::MappingDesc>& mapper,
            T_KernelInput... kernelInput);
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
