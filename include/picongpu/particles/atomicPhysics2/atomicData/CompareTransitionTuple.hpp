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
#include "picongpu/particles/atomicPhysics2/atomicData/GetStateFromTransitionTuple.hpp"

#include <cstdint>
#include <tuple>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
            {
                /** comparison functor in between transition tuples
                 *
                 * @tparam T_Number data type used for numbers
                 * @tparam T_idx data type used for condigNumbers
                 * @tparam orderByLowerState true: order by by lower , false: upper state
                 */
                template<typename T_Value, typename T_Idx, bool orderByLowerState>
                class CompareTransitionTupel
                {

                public:
                    template< typename T_Tuple >
                    bool operator()(T_Tuple& tuple_1, T_Tuple& tuple_2)
                    {
                        T_Idx lowerState_1 = getLowerStateConfigNumber<T_Idx, T_Value>(tuple_1);
                        T_Idx lowerState_2 = getLowerStateConfigNumber<T_Idx, T_Value>(tuple_2);

                        T_Idx upperState_1 = getUpperStateConfigNumber<T_Idx, T_Value>(tuple_1);
                        T_Idx upperState_2 = getUpperStateConfigNumber<T_Idx, T_Value>(tuple_2);

                        if constexpr(orderByLowerState)
                            return (
                                (lowerState_1 < lowerState_2)
                                or ((lowerState_1 == lowerState_2) and (upperState_1 < upperState_2)));
                        else
                            return (
                                (upperState_1 < upperState_2)
                                or ((upperState_1 == upperState_2) and (lowerState_1 < lowerState_2)));
                    }
                };
            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
