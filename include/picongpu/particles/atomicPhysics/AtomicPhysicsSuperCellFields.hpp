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
#include "picongpu/particles/atomicPhysics/ParticleType.hpp"
#include "picongpu/particles/atomicPhysics/electronDistribution/LocalHistogramField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/ElectronHistogramOverSubscribedField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/FieldEnergyUseCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/FoundUnboundIonField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/RejectionProbabilityCacheField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeRemainingField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/TimeStepField.hpp"
#include "picongpu/particles/atomicPhysics/stage/CreateRateCacheField.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/meta/ForEach.hpp>

namespace picongpu::particles::atomicPhysics
{
    struct AtomicPhysicsSuperCellFields
    {
        using ListAtomicPhysicsSpecies = particles::atomicPhysics::traits::
            FilterByParticleType_t<VectorAllSpecies, picongpu::particles::atomicPhysics::Tags::Ion>;

        //! create all superCell fields required by the atomicPhysics core loops, are stored in dataConnector
        HINLINE static void create(DataConnector& dataConnector, picongpu::MappingDesc const mappingDesc)
        {
            // local electron interaction histograms
            auto localSuperCellElectronHistogramField = std::make_unique<electronDistribution::LocalHistogramField<
                // defined/set in atomicPhysics.param
                picongpu::atomicPhysics::ElectronHistogram,
                // defined in memory.param
                picongpu::MappingDesc>>(mappingDesc, "Electron");
            dataConnector.consume(std::move(localSuperCellElectronHistogramField));

            ///@todo repeat for "Photons" once implemented, Brian Marre, 2022

            // local rate cache, create in pre-stage call for each species
            pmacc::meta::ForEach<
                ListAtomicPhysicsSpecies,
                particles::atomicPhysics::stage::CreateRateCacheField<boost::mpl::_1>>
                ForEachIonSpeciesCreateRateCacheField;
            ForEachIonSpeciesCreateRateCacheField(dataConnector, mappingDesc);

            // local time remaining field
            auto superCellTimeRemainingField
                = std::make_unique<localHelperFields::TimeRemainingField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(superCellTimeRemainingField));

            // local time step field
            auto superCellTimeStepField
                = std::make_unique<localHelperFields::TimeStepField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(superCellTimeStepField));

            // local electron histogram over subscribed switch
            auto superCellElectronHistogramOverSubscribedField
                = std::make_unique<localHelperFields::ElectronHistogramOverSubscribedField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(superCellElectronHistogramOverSubscribedField));

            // local storage for foundUnboundIon switch
            auto foundUnboundIonField
                = std::make_unique<localHelperFields::FoundUnboundIonField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(foundUnboundIonField));

            // local rejection probability for each over-subscribed bin of the electron histogram
            auto superCellRejectionProbabilityCacheField
                = std::make_unique<localHelperFields::RejectionProbabilityCacheField<picongpu::MappingDesc>>(
                    mappingDesc);
            dataConnector.consume(std::move(superCellRejectionProbabilityCacheField));

            // local field energy use cache
            auto superCellFieldEnergyUseCacheField
                = std::make_unique<localHelperFields::FieldEnergyUseCacheField<picongpu::MappingDesc>>(mappingDesc);
            dataConnector.consume(std::move(superCellFieldEnergyUseCacheField));
        }
    };
} // namespace picongpu::particles::atomicPhysics
