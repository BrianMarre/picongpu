/* Copyright 2022 Rene Widera, Sergei Bastrakov, Brian Marre
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

#include "picongpu/particles/atomicPhysics2/BinElectrons.hpp"

#include <pmacc/meta/ForEach.hpp>


/** @file
 *
 * This file implements the Atomic Physics stage of the PIC-loop.
 *
 * One instance of this class AtomicPhysics is stored as a protected member of the
 * MySimulation class.
 */

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** atomic physics stage
             *
             * one instance of this class is initialized and it's operator() called for every time step
             */
            struct AtomicPhysics2
            {
            private:
                /** list of all species of macro particles with flag isElectron
                 *
                 * as defined in species.param, is list of types
                 */
                using SpeciesRepresentingElectrons =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, isElectron<>>::type;

                //! kernel to be called for each species
                pmacc::meta::ForEach<SpeciesRepresentingElectrons, particles::atomicPhysics2::BinElectrons<bmpl::_1>>
                    ForEachElectronSpeciesBinElectrons;

                //! reset the histogram on device side
                static void resetHistograms(MappingDesc const mappingDesc)
                {
                    pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

                    auto& localElectronHistogramField
                        = *dc.get<particles::atomicPhysics2::electronDistribution::LocalHistogramField<
                            picongpu::atomicPhysics2::ElectronHistogram,
                            picongpu::MappingDesc>>("Electron_localHistogramField", true);

                    localElectronHistogramField.getDeviceBuffer().setValue(
                        picongpu::atomicPhysics2::ElectronHistogram());
                }

                //! set local timeRemaining to PIC-time step
                static void setTimeRemaining(MappingDesc const mappingDesc)
                {
                    pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();

                    auto& localTimeRemainingField
                        = *dc.get<particles::atomicPhysics2::LocalTimeRemainingField<picongpu::MappingDesc>>(
                            "LocalTimeRemainingField",
                            true);

                    localTimeRemainingField.getDeviceBuffer().setValue(picongpu::SI::DELTA_T_SI);
                }

            public:
                AtomicPhysics2() = default;

                //! calls the substages
                void operator()(MappingDesc const mappingDesc)
                {
                    // set timeRemaining
                    setTimeRemaining(mappingDesc);

                    // bin Electrons
                    // reset Histogram
                    resetHistograms(mappingDesc);

                    // actual binning
                    ForEachElectronSpeciesBinElectrons(mappingDesc);

                    //
                }

            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
