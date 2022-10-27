/* Copyright 2022 Brian Marre
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

#include "picongpu/particles/atomicPhysics2/atomicData/AtomicTuples.def"
#include "picongpu/particles/atomicPhysics2/atomicData/GetStateFromTransitionTupel.hpp"

#include <tupel>
#include <cstdint>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
            {
                /** comparison functor in between transition tupels
                 *
                 * @tparam T_Number data type used for numbers
                 * @tparam T_idx data type used for condigNumbers
                 * @tparam orderByLowerState true: order by by lower , false: upper state
                 */
                template<typename T_Value, typename T_Idx, bool orderByLowerState>
                class CompareTransitionTupel
                {
                    using S_BoundBoundTransitionTuple = BoundBoundTransitionTuple<T_Value, Idx>;
                    using S_BoundFreeTransitionTuple = BoundFreeTransitionTuple<T_Value, Idx>;
                    using S_AutonomousTransitionTuple = AutonomousTransitionTuple<T_Value, Idx>;

                    bool operator(S_BoundBoundTransitionTuple tuple1_1, S_BoundBoundTransitionTuple tuple_2)
                    {
                        T_Idx lowerState_1 = getLowerState<T_Idx, T_Value>(tuple1_1);
                        T_Idx lowerState_2 = getLowerState<T_Idx, T_Value>(tuple1_2);

                        T_Idx upperState_1 = getUpperState<T_Idx, T_Value>(tuple1_1);
                        T_Idx upperState_2 = getUpperState<T_Idx, T_Value>(tuple1_2);

                        if constexpr(orderByLowerState)
                            return ( (lowerState_1 < lowerState_2) or ( (lowerState_1 == lowerState_2) and (upperState_1 < upperState_2) ) );
                        else
                            return ( (upperState_1 < upperState_2) or ( (upperState_1 == upperState_2) and (lowerState_1 < lowerState_2) ) );
                    }

                };
            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
