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

/** @file interface definition for all matter ionization potential depression(IPD) implementations
 *
 * An matter IPD implementation is a wrapper for one or more MatterIPDModel that provides a common interface for IPD
 *  calculation due to matter interactions, i.e. ion/electron background, for use by the atomicPhysics stage,
 *  independent of the actual IPDModel used.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/mappings/kernel/AreaMapping.hpp>

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    struct MatterIPDImplementation
    {
        //! create all HelperFields required by the IPD model
        ALPAKA_FN_HOST static void createHelperFields();

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
        HINLINE static void calculateIPDInput(picongpu::MappingDesc const mappingDesc);

        /** calculate ionization potential depression
         *
         * @param superCellFieldIdx index of superCell in superCellField(without guards)
         * @param ipdInput to ipd calculation
         *
         * @return unit: eV, not weighted
         */
        template<typename... T_IPDInput>
        HDINLINE static float_X calculateIPD(
            pmacc::DataSpace<simDim> const superCellFieldIdx,
            T_IPDInput const... ipdInput);

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
