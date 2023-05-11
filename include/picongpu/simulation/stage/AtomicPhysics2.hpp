/* Copyright 2022-2023 Brian Marre
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
// need picongpu::simDim from picongpu/param/dimension.param

#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalAllMacroIonsAcceptedField.hpp"
#include "picongpu/particles/atomicPhysics2/localHelperFields/LocalTimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics2/stage/AcceptTransitionTest.stage"
#include "picongpu/particles/atomicPhysics2/stage/BinElectrons.stage"
#include "picongpu/particles/atomicPhysics2/stage/CalculateStepLength.stage"
#include "picongpu/particles/atomicPhysics2/stage/CheckForAcceptance.stage"
#include "picongpu/particles/atomicPhysics2/stage/CheckForOverSubscription.stage"
#include "picongpu/particles/atomicPhysics2/stage/ChooseTransition.stage"
#include "picongpu/particles/atomicPhysics2/stage/DecelerateElectrons.stage"
#include "picongpu/particles/atomicPhysics2/stage/ExtractTransitionCollectionIndex.stage"
#include "picongpu/particles/atomicPhysics2/stage/FillLocalRateCache.stage"
#include "picongpu/particles/atomicPhysics2/stage/RecordChanges.stage"
#include "picongpu/particles/atomicPhysics2/stage/RecordSuggestedChanges.stage"
#include "picongpu/particles/atomicPhysics2/stage/ResetAcceptedStatus.stage"
#include "picongpu/particles/atomicPhysics2/stage/ResetLocalRateCache.stage"
#include "picongpu/particles/atomicPhysics2/stage/ResetLocalTimeStepField.stage"
#include "picongpu/particles/atomicPhysics2/stage/RollForOverSubscription.stage"
#include "picongpu/particles/atomicPhysics2/stage/SpawnIonizationElectrons.stage"
#include "picongpu/particles/atomicPhysics2/stage/UpdateTimeRemaining.stage"

#include <pmacc/device/Reduce.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>
#include <string>

/** @file implements the Atomic Physics stage of the PIC-loop.
 *
 * One instance of this class AtomicPhysics is stored as a protected member of the
 * MySimulation class.
 *
 * partially based upon a previous version built by Sergei Bastrakov and Brian Marre
 */

namespace picongpu::simulation::stage
{
    /** atomic physics stage
     *
     * excited atomic state and ionization dynamics
     *
     * one instance of this class is initialized and it's operator() called for every time step
     *
     */
    struct AtomicPhysics2
    {
    private:
        // linearized dataBox of SuperCellField
        template<typename T_Field>
        using S_LinearizedBox = DataBoxDim1Access<typename T_Field::DataBoxType>;

        using S_OverSubscribedField
            = picongpu::particles::atomicPhysics2::localHelperFields ::LocalElectronHistogramOverSubscribedField<
                picongpu::MappingDesc>;
        using S_AllIonsAcceptedField
            = picongpu::particles::atomicPhysics2::localHelperFields ::LocalAllMacroIonsAcceptedField<
                picongpu::MappingDesc>;
        using S_TimeRemainingField
            = particles::atomicPhysics2::localHelperFields ::LocalTimeRemainingField<picongpu::MappingDesc>;

        /** list of all species of macro particles with flag isAtomicPhysicsElectron
         *
         * as defined in species.param, is list of types
         */
        using SpeciesRepresentingElectrons =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isAtomicPhysicsElectron<>>::type;
        /** list of all species of macro particles with atomicPhysics input data
         *
         * as defined in species.param, is list of types
         * @todo use different Flag?, Brian Marre, 2023
         */
        using SpeciesRepresentingIons =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, atomicDataType<>>::type;

        //! set local timeRemaining to PIC-time step
        HINLINE static void setTimeRemaining()
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localTimeRemainingField = *dc.get<S_TimeRemainingField>("LocalTimeRemainingField");

            localTimeRemainingField.getDeviceBuffer().setValue(DELTA_T); // UNIT_TIME
        }

        //! reset local allMacroIonsAccepted switch to ture
        HINLINE static void resetAllMacroIonsAcceptedField()
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localAllIonsAcceptedField = *dc.get<S_AllIonsAcceptedField>("LocalAllMacroIonsAcceptedField");

            localAllIonsAcceptedField.getDeviceBuffer().setValue(true);
        }

        //! reset the histogram on device side
        HINLINE static void resetHistograms()
        {
            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            auto& localElectronHistogramField
                = *dc.get<particles::atomicPhysics2::electronDistribution::
                              LocalHistogramField<picongpu::atomicPhysics2::ElectronHistogram, picongpu::MappingDesc>>(
                    "Electron_localHistogramField");

            localElectronHistogramField.getDeviceBuffer().setValue(picongpu::atomicPhysics2::ElectronHistogram());
        }

    public:
        AtomicPhysics2() = default;

        //! atomic physics stage sub-stage calls
        void operator()(picongpu::MappingDesc const mappingDesc, uint32_t const currentStep) const
        {
            //! reset macro particle attribute accepted to false for each ion species
            using ForEachIonSpeciesResetAcceptedStatus = pmacc::meta::
                ForEach<SpeciesRepresentingIons, particles::atomicPhysics2::stage::ResetAcceptedStatus<boost::mpl::_1>>;
            //! bin electrons sub stage call for each electron species
            using ForEachElectronSpeciesBinElectrons = pmacc::meta::
                ForEach<SpeciesRepresentingElectrons, particles::atomicPhysics2::stage::BinElectrons<boost::mpl::_1>>;
            //! reset localRateCacheField sub stage for each ion species
            using ForEachIonSpeciesResetLocalRateCache = pmacc::meta::
                ForEach<SpeciesRepresentingIons, particles::atomicPhysics2::stage::ResetLocalRateCache<boost::mpl::_1>>;
            //! fill rate cache with diagonal elements of rate matrix
            using ForEachIonSpeciesFillLocalRateCache = pmacc::meta::
                ForEach<SpeciesRepresentingIons, particles::atomicPhysics2::stage::FillLocalRateCache<boost::mpl::_1>>;
            //! calculate local atomicPhysics time step length
            using ForEachIonSpeciesCalculateStepLength = pmacc::meta::
                ForEach<SpeciesRepresentingIons, particles::atomicPhysics2::stage::CalculateStepLength<boost::mpl::_1>>;
            //! chooseTransition for every macro-ion
            using ForEachIonSpeciesChooseTransition = pmacc::meta::
                ForEach<SpeciesRepresentingIons, particles::atomicPhysics2::stage::ChooseTransition<boost::mpl::_1>>;
            //! extract transitionCollectionIndex
            using ForEachIonSpeciesExtractTransitionCollectionIndex = pmacc::meta::ForEach<
                SpeciesRepresentingIons,
                particles::atomicPhysics2::stage::ExtractTransitionCollectionIndex<boost::mpl::_1>>;
            //! try to accept transitions
            using ForEachIonSpeciesDoAcceptTransitionTest = pmacc::meta::
                ForEach<SpeciesRepresentingIons, particles::atomicPhysics2::stage::AcceptTransitionTest<boost::mpl::_1>>;
            //! record suggested changes
            using ForEachIonSpeciesRecordSuggestedChanges = pmacc::meta::
                ForEach<SpeciesRepresentingIons, particles::atomicPhysics2::stage::RecordSuggestedChanges<boost::mpl::_1>>;
            //! roll for rejection of transitions due to over subscription
            using ForEachIonSpeciesRollForOverSubscription = pmacc::meta::
                ForEach<SpeciesRepresentingIons, particles::atomicPhysics2::stage::RollForOverSubscription<boost::mpl::_1>>;
            //! check for acceptance of a transition by all ions
            using ForEachIonSpeciesCheckForAcceptance = pmacc::meta::
                ForEach<SpeciesRepresentingIons, particles::atomicPhysics2::stage::CheckForAcceptance<boost::mpl::_1>>;
            //! record delta energy for all transitions
            using ForEachIonSpeciesRecordChanges = pmacc::meta::ForEach<SpeciesRepresentingIons, particles::atomicPhysics2::stage::RecordChanges<boost::mpl::_1>>;
            //! decelerate all electrons according to their bin delta energy
            using ForEachElectronSpeciesDecelerateElectrons = pmacc::meta::ForEach<
                SpeciesRepresentingElectrons,
                particles::atomicPhysics2::stage::DecelerateElectrons<boost::mpl::_1>>;
            //! spawn ionization created macro electrons due to atomicPhysics processes
            using ForEachIonSpeciesSpawnIonizationElectrons = pmacc::meta::ForEach<
                SpeciesRepresentingIons,
                particles::atomicPhysics2::stage::SpawnIonizationElectrons<boost::mpl::_1>>;

            pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

            // TimeRemainingSuperCellField
            auto& localTimeRemainingField = *dc.get<S_TimeRemainingField>("LocalTimeRemainingField");
            DataSpace<picongpu::simDim> const fieldGridLayoutTimeRemaining
                = localTimeRemainingField.getGridLayout().getDataSpaceWithoutGuarding();

            // AllMacroIonsAcceptedSuperCellField
            auto& localAllIonsAcceptedField = *dc.get<S_AllIonsAcceptedField>("LocalAllMacroIonsAcceptedField");
            DataSpace<picongpu::simDim> const fieldGridLayoutAllIonsAccepted
                = localAllIonsAcceptedField.getGridLayout().getDataSpaceWithoutGuarding();

            // ElectronHistogramOverSubscribedSuperCellField
            auto& localElectronHistogramOverSubscribedField
                = *dc.get<S_OverSubscribedField>("LocalElectronHistogramOverSubscribedField");
            DataSpace<picongpu::simDim> const fieldGridLayoutOverSubscription
                = localElectronHistogramOverSubscribedField.getGridLayout().getDataSpaceWithoutGuarding();

            /// @todo find better way than hard code old value, Brian Marre, 2023
            pmacc::device::Reduce deviceLocalReduce = pmacc::device::Reduce(static_cast<uint32_t>(1200u));

            setTimeRemaining(); // = (Delta t)_PIC

            // atomicPhysics sub-stepping loop, ends when timeRemaining<=0._X
            while(true)
            {
                // particle[accepted_] = false, in each macro ion
                ForEachIonSpeciesResetAcceptedStatus{}(mappingDesc);
                resetHistograms();
                ForEachElectronSpeciesBinElectrons{}(mappingDesc);
                // timeStep = localTimeRemaining
                picongpu::particles::atomicPhysics2::stage::ResetLocalTimeStepField()(mappingDesc);
                ForEachIonSpeciesResetLocalRateCache{}();
                // R_ii = -(sum of rates of all transitions from state i)
                ForEachIonSpeciesFillLocalRateCache{}(mappingDesc);
                // min(1/(-R_ii)) * alpha
                ForEachIonSpeciesCalculateStepLength{}(mappingDesc);

                // chooseTransition loop, ends when all ion[accepted_] = true
                while(true)
                {
                    // randomly roll transition for each not yet accepted macro ion
                    ForEachIonSpeciesChooseTransition{}(mappingDesc, currentStep);
                    ForEachIonSpeciesExtractTransitionCollectionIndex{}(mappingDesc, currentStep);
                    ForEachIonSpeciesDoAcceptTransitionTest{}(mappingDesc, currentStep);
                    ForEachIonSpeciesRecordSuggestedChanges{}(mappingDesc);

                    // reject overSubscription loop, ends when no histogram bin oversubscribed
                    while(true)
                    {
                        // check bins for over subscription --> localElectronHistogramOverSubscribedField
                        picongpu::particles::atomicPhysics2::stage::CheckForOverSubscription()(mappingDesc);

                        auto linearizedOverSubscribedBox = S_LinearizedBox<S_OverSubscribedField>(
                            localElectronHistogramOverSubscribedField.getDeviceDataBox(),
                            fieldGridLayoutOverSubscription);

                        if(!static_cast<bool>(deviceLocalReduce(
                               pmacc::math::operation::Or(),
                               linearizedOverSubscribedBox,
                               fieldGridLayoutOverSubscription.productOfComponents())))
                        {
                            /* no superCell electron histogram marked as over subscribed in
                             *  localElectronHistogramOverSubscribedField */
                            break;
                        }
                        // at least one superCell electron histogram over subscribed
                        ForEachIonSpeciesRollForOverSubscription{}(mappingDesc, currentStep);
                    } // end reject overSubscription loop

                    // check all macro-ions accepted --> localAllIonsAcceptedField
                    resetAllMacroIonsAcceptedField(); // local field, NOT macro ion particle attribute
                    ForEachIonSpeciesCheckForAcceptance{}(mappingDesc);

                    auto linearizedAllAcceptedBox = S_LinearizedBox<S_AllIonsAcceptedField>(
                        localAllIonsAcceptedField.getDeviceDataBox(),
                        fieldGridLayoutAllIonsAccepted);

                    // all Ions accepted?
                    if(static_cast<bool>(deviceLocalReduce(
                           pmacc::math::operation::And(),
                           linearizedAllAcceptedBox,
                           fieldGridLayoutAllIonsAccepted.productOfComponents())))
                    {
                        break;
                    }
                } // end chooseTransition loop

                // record changes electron spectrum
                ForEachIonSpeciesRecordChanges{}(mappingDesc);
                ForEachElectronSpeciesDecelerateElectrons{}(mappingDesc);
                ForEachIonSpeciesSpawnIonizationElectrons{}(mappingDesc, currentStep);

                // timeRemaining -= timeStep
                picongpu::particles::atomicPhysics2::stage::UpdateTimeRemaining()(mappingDesc);

                auto linearizedTimeRemainingBox = S_LinearizedBox<S_TimeRemainingField>(
                    localTimeRemainingField.getDeviceDataBox(),
                    fieldGridLayoutTimeRemaining);

                // timeRemaining <= 0? in all local superCells?
                if(deviceLocalReduce(
                       pmacc::math::operation::Max(),
                       linearizedTimeRemainingBox,
                       fieldGridLayoutTimeRemaining.productOfComponents())
                   <= 0._X)
                {
                    break;
                }
            } // end atomicPhysics sub-stepping loop
        }
    };
} // namespace picongpu::simulation::stage
