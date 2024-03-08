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

ApplyPressureIonization

    namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression::stage
{
    //! short hand for IPD namespace
    namespace s_IPD = picongpu::particles::atomicPhysics2::ionizationPotentialDepression;

    //! IPD sub-stage for performing ApplyPressureIonization kernel call for one Ion Species
    template<typename T_IonSpecies>
    struct ApplyPressureIonization
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_ParticleSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        //! resolved type of electron species to spawn upon ionization
        using IonizationElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<
            VectorAllSpecies,
            typename picongpu::traits::GetIonizationElectronSpecies<T_IonSpecies>::type>;

        using AtomicDataType = typename picongpu::traits::GetAtomicDataType<T_IonSpecies>::type;

        // full local domain, no guards
        pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
        pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
        pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg<T_IonSpecies::FrameType::frameSize>();

        auto& localTimeRemainingField = *dc.get<
            picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeRemainingField<picongpu::MappingDesc>>(
            "LocalTimeRemainingField");

        auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());
        auto& electrons = *dc.get<IonizationElectronSpecies>(IonizationElectronSpecies::FrameType::getName());

        auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

        auto& localDebyeLengthField = *dc.get<
            picongpu::particles::atomicPhysics2::localHelperFields::LocalDebyeLengthField<picongpu::MappingDesc>>(
            "LocalTimeRemainingField");


        // macro for call of kernel on every superCell, see pull request #4321
        PMACC_LOCKSTEP_KERNEL(ApplyPressureIonization<StewartPyattIPD>(), workerCfg)
        (mapper.getGridDim())(
            mapper,
            localTimeRemainingField.getDeviceDataBox(),
            ions.getDeviceParticleBox(),
            electrons.getDeviceParticleBox(),
            localTimeRemainingField.getDeviceDataBox(),
            atomicData.getChargeStateDataDataBox</*on device*/ false>(),
            atomicData.getAtomicStateDataDataBox</*on device*/ false>(),
            atomicData.getPressureIonizationStateDataBox</*on device*/ false>(), );

        // no need to call fillAllGaps, since we do not leave any gaps

        // debug call
        if constexpr(picongpu::atomicPhysics2::debug::kernel::spawnIonizationElectrons::
                         ELECTRON_PARTICLE_BOX_FILL_GAPS)
            electrons.fillAllGaps();


    } // namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression::stage
