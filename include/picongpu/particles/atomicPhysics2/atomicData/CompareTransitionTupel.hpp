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

#include <tupel>

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
                template<typename T_Number, typename T_Idx, bool orderByLowerState>
                class CompareTransitionTupel
                {
                    using S_BoundBoundTransitionTuple = BoundBoundTransitionTuple<TypeValue, Idx>;
                    using S_BoundFreeTransitionTuple = BoundFreeTransitionTuple<TypeValue, Idx>;
                    using S_AutonomousTransitionTuple = AutonomousTransitionTuple<TypeValue, Idx>;

                    bool operator(S_BoundBoundTransitionTuple tuple1_1, S_BoundBoundTransitionTuple tuple_2)
                    {
                        if constexpr(orderByLowerState)
                            return std::get<7>(tuple1_1) < std::get<7>(tupel_2);
                        else
                            return std::get<8>(tuple1_1) < std::get<8>(tupel_2);
                    }

                    bool operator(S_BoundFreeTransitionTuple tuple1_1, S_BoundFreeTransitionTuple tuple_2)
                    {
                        if constexpr(orderByLowerState)
                            return std::get<8>(tuple1_1) < std::get<8>(tupel_2);
                        else
                            return std::get<9>(tuple1_1) < std::get<9>(tupel_2);
                    }

                    bool operator(S_AutonomousTransitionTuple tuple1_1, S_AutonomousTransitionTuple tuple_2)
                    {
                        if constexpr(orderByLowerState)
                            return std::get<1>(tuple1_1) < std::get<1>(tupel_2);
                        else
                            return std::get<2>(tuple1_1) < std::get<2>(tupel_2);
                    }
                };
            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
