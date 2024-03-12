/* Copyright 2023 Brian Marre
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

//! @file deduceTransitionCollectionIndex sub-stage of atomicPhysics

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/enums/TransitionDirection.hpp"
#include "picongpu/particles/atomicPhysics2/enums/TransitionOrdering.hpp"
#include "picongpu/particles/atomicPhysics2/kernel/ChooseTransition_Autonomous.kernel"
#include "picongpu/particles/atomicPhysics2/kernel/ChooseTransition_BoundBound.kernel"
#include "picongpu/particles/atomicPhysics2/kernel/ChooseTransition_BoundFree.kernel"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalTimeRemainingField.hpp"

namespace picongpu::particles::atomicPhysics2::stage
{
    namespace s_enums = picongpu::particles::atomicPhysics2::enums;

    /** atomic physics sub-stage choosing the specific transition from the previously chosen transitionType
     *    for each macro-ion of the given species
     *
     * @attention assumes that the the chooseTransitionType kernel has been completed already
     *
     * @tparam T_IonSpecies ion species type
     */
    template<typename T_IonSpecies>
    struct ChooseTransition
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        using DistributionFloat = pmacc::random::distributions::Uniform<float_X>;
        using RngFactoryFloat = particles::functor::misc::Rng<DistributionFloat>;

        //! call of kernel for every superCell
        HINLINE void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
        {
            // full local domain, no guards
            pmacc::AreaMapping<CORE + BORDER, MappingDesc> mapper(mappingDesc);
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
            pmacc::lockstep::WorkerCfg workerCfg = pmacc::lockstep::makeWorkerCfg<IonSpecies::FrameType::frameSize>();

            using AtomicDataType = typename picongpu::traits::GetAtomicDataType<IonSpecies>::type;
            auto& atomicData = *dc.get<AtomicDataType>(IonSpecies::FrameType::getName() + "_atomicData");

            auto& localTimeRemainingField
                = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::LocalTimeRemainingField<
                    picongpu::MappingDesc>>("LocalTimeRemainingField");
            auto& localElectronHistogramField
                = *dc.get<picongpu::particles::atomicPhysics2::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics2::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");
            using RateCache = typename picongpu::particles::atomicPhysics2::localHelperFields::
                LocalRateCacheField<picongpu::MappingDesc, IonSpecies>::entryType;
            auto& localRateCacheField = *dc.get<picongpu::particles::atomicPhysics2::localHelperFields::
                                                    LocalRateCacheField<picongpu::MappingDesc, IonSpecies>>(
                IonSpecies::FrameType::getName() + "_localRateCacheField");

            auto& ions = *dc.get<IonSpecies>(IonSpecies::FrameType::getName());
            RngFactoryFloat rngFactoryFloat = RngFactoryFloat{currentStep};

            // no-change transitions are already accepted by chooseTransitionTypeKernel

            // bound-bound(upward) transitions
            if constexpr(AtomicDataType::switchElectronicExcitation)
            {
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel::ChooseTransitionKernel_BoundBound<
                        picongpu::atomicPhysics2::ElectronHistogram,
                        AtomicDataType::ConfigNumber::numberLevels,
                        s_enums::TransitionDirection::upward,
                        AtomicDataType::switchElectronicExcitation,
                        AtomicDataType::switchElectronicDeexcitation,
                        AtomicDataType::switchSpontaneousDeexcitation>(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    rngFactoryFloat,
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundBoundNumberTransitionsDataBox<false>(),
                    atomicData.template getBoundBoundStartIndexBlockDataBox<false>(),
                    atomicData
                        .template getBoundBoundTransitionDataBox<false, s_enums::TransitionOrdering::byLowerState>(),
                    localTimeRemainingField.getDeviceDataBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    localRateCacheField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox());
            }

            // bound-bound(downward) transitions
            if constexpr(AtomicDataType::switchElectronicDeexcitation || AtomicDataType::switchSpontaneousDeexcitation)
            {
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel::ChooseTransitionKernel_BoundBound<
                        picongpu::atomicPhysics2::ElectronHistogram,
                        AtomicDataType::ConfigNumber::numberLevels,
                        s_enums::TransitionDirection::downward,
                        AtomicDataType::switchElectronicExcitation,
                        AtomicDataType::switchElectronicDeexcitation,
                        AtomicDataType::switchSpontaneousDeexcitation>(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    rngFactoryFloat,
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundBoundNumberTransitionsDataBox<false>(),
                    atomicData.template getBoundBoundStartIndexBlockDataBox<false>(),
                    atomicData
                        .template getBoundBoundTransitionDataBox<false, s_enums::TransitionOrdering::byUpperState>(),
                    localTimeRemainingField.getDeviceDataBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    localRateCacheField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox());
            }

            // bound-free(upward) transitions
            if constexpr(AtomicDataType::switchElectronicIonization || AtomicDataType::switchFieldIonization)
            {
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel::ChooseTransitionKernel_BoundFree<
                        picongpu::atomicPhysics2::ElectronHistogram,
                        AtomicDataType::ConfigNumber::numberLevels,
                        AtomicDataType::switchElectronicIonization,
                        AtomicDataType::switchFieldIonization>(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    rngFactoryFloat,
                    atomicData.template getChargeStateDataDataBox<false>(),
                    atomicData.template getAtomicStateDataDataBox<false>(),
                    atomicData.template getBoundFreeNumberTransitionsDataBox<false>(),
                    atomicData.template getBoundFreeStartIndexBlockDataBox<false>(),
                    atomicData
                        .template getBoundFreeTransitionDataBox<false, s_enums::TransitionOrdering::byLowerState>(),
                    localTimeRemainingField.getDeviceDataBox(),
                    localElectronHistogramField.getDeviceDataBox(),
                    localRateCacheField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox());
            }

            // autonomous transitions
            if constexpr(AtomicDataType::switchAutonomousIonization)
            {
                PMACC_LOCKSTEP_KERNEL(
                    picongpu::particles::atomicPhysics2::kernel::ChooseTransitionKernel_Autonomous(),
                    workerCfg)
                (mapper.getGridDim())(
                    mapper,
                    rngFactoryFloat,
                    atomicData.template getAutonomousNumberTransitionsDataBox<false>(),
                    atomicData.template getAutonomousStartIndexBlockDataBox<false>(),
                    atomicData
                        .template getAutonomousTransitionDataBox<false, s_enums::TransitionOrdering::byUpperState>(),
                    localTimeRemainingField.getDeviceDataBox(),
                    localRateCacheField.getDeviceDataBox(),
                    ions.getDeviceParticlesBox());
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics2::stage
