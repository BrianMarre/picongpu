/* Copyright 2022 Brian Marre, Rene Widera
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

#include <pmacc/particles/memory/frames/Frame.hpp>
#include <pmacc/static_assert.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>


namespace picongpu::particles::atomicPhysics2
{
    /** pre-simulation stage for loading the user provided atomic input data
     *
     * @tparam T_IonSpecies species for which to call the functor
     */
    template<typename T_IonSpecies>
    struct LoadAtomicInputData
    {
        // might be alias, from here on out no more
        //! resolved type of alias T_IonSpecies
        using IonSpecies = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_IonSpecies>;

        HINLINE void operator()(DataConnector& dataConnector)
        {
            /// @todo move to trait functor?, Brian Marre, 2022
            using FrameType = typename T_IonSpecies::FrameType;

            using hasAtomicInputData = typename HasFlag<FrameType, isIonWithAtomicPhysicsInputData<>>::type;

            /* throw static assert if species has no atomicData Type */
            PMACC_CASSERT_MSG(This_species_has_no_atomicData_Type, hasAtomicNumbers::value == true);

            using AtomicDataType = typename GetFlagType<FrameType, isIonWithAtomicPhysicsInputData<>>::type;
            using type = typename pmacc::traits::Resolve<AtomicDataType>::type;

            using S_AtomicData =
        }
    };

} // namespace picongpu::particles::atomicPhysics2
