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

#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/ApplyIPDIonization.kernel"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalFoundUnboundIonField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/LocalTimeRemainingField.hpp"
#include "picongpu/particles/traits/GetAtomicDataType.hpp"
#include "picongpu/particles/traits/GetIonizationElectronSpecies.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    /** IPD sub-stage for applying the effects of ionization due to ionization potential depression(IPD)
     *
     * calls the ApplyIPDIonization kernel call for one Ion Species for each local superCell
     *
     * @tparam T_IonSpecies ion species to apply IPD ionization to
     * @tparam T_IPDImplementation ionization potential depression implementation to use
     *
     */
    template<typename T_IonSpecies, typename T_IPDImplementation>
    struct ApplyIPDIonization
    {
        /// @todo implement version for non atomicPhysics data species
        stopper;

        // might be alias, from here on out no more
        //! resolved type of alias T_ParticleSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! call of kernel for every superCell
        template<typename... T_IPDInputBoxes>
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc, T_IPDInputBoxes... ipdInputBoxes) const
        {
            //! resolved type of electron species to spawn upon ionization
            using IonizationElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<
                VectorAllSpecies,
                typename picongpu::traits::GetIonizationElectronSpecies<T_IonSpecies>::type>;
            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<T_IonSpecies>::type;

            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localTimeRemainingField
                = *dc.get<atomicPhysics::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
                    "LocalTimeRemainingField");
            auto& localFoundUnboundIonField
                = *dc.get<atomicPhysics::localHelperFields::LocalFoundUnboundIonField<picongpu::MappingDesc>>(
                    "LocalFoundUnboundIonField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());
            auto& electrons = *dc.get<IonizationElectronSpecies>(IonizationElectronSpecies::FrameType::getName());

            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");
            auto idProvider = dc.get<IdProvider>("globalId");

            // macro for call of kernel on every superCell, see pull request #4321
            PMACC_LOCKSTEP_KERNEL(s_IPD::kernel::ApplyIPDIonizationKernel<T_IPDImplementation>())
                .config(mapper.getGridDim(), ions)(
                    mapper,
                    idProvider->getDeviceGenerator(),
                    ions.getDeviceParticlesBox(),
                    electrons.getDeviceParticlesBox(),
                    localTimeRemainingField.getDeviceDataBox(),
                    localFoundUnboundIonField.getDeviceDataBox(),
                    atomicData.template getChargeStateDataDataBox</*on device*/ false>(),
                    atomicData.template getAtomicStateDataDataBox</*on device*/ false>(),
                    atomicData.template getIPDIonizationStateDataBox</*on device*/ false>(),
                    ipdInputBoxes...);

            // no need to call fillAllGaps, since we do not leave any gaps
            // debug call
            if constexpr(picongpu::atomicPhysics::debug::kernel::applyIPDIonization::ELECTRON_PARTICLE_BOX_FILL_GAPS)
                electrons.fillAllGaps();
        }
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
