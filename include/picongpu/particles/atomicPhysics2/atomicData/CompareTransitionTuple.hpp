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

#include "picongpu/particles/atomicPhysics2/atomicData/AtomicTuples.def"
#include "picongpu/particles/atomicPhysics2/atomicData/GetStateFromTransitionTuple.hpp"

#include "picongpu/particles/atomicPhysics2/stateRepresentation/ConfigNumber.hpp"

#include <cstdint>
#include <tuple>
#include <stdexcept>
#include <string>

namespace picongpu::particles::atomicPhysics2::atomicData
{
    /** comparison functor in between transition tuples
     *
     * transitions are ordered primarily ascending by lower/upper charge state,
     *  secondary ascending by lower/upper atomicConfigNumber,
     *  tertiary ascending by upper/lower charge state,
     *  quartary ascending by upper/lower atomicConfigNumber
     *
     * @tparam T_Number data type used for numbers
     * @tparam T_ConfigNumber dataType used for storage of configNumber
     * @tparam orderByLowerState true=^=order by by lower , false=^=upper state
     */
    template<
        typename T_Value,
        typename T_ConfigNumber,
        bool orderByLowerState>
    class CompareTransitionTupel
    {
    public:
        template<typename T_Tuple>
        bool operator()(T_Tuple& tuple_1, T_Tuple& tuple_2)
        {
            using Idx = typename T_ConfigNumber::DataType;

            Idx lowerAtomicState_1 = getLowerStateConfigNumber<Idx, T_Value>(tuple_1);
            Idx lowerAtomicState_2 = getLowerStateConfigNumber<Idx, T_Value>(tuple_2);
            uint8_t lowerChargeState_1 = T_ConfigNumber::getIonizationState(
                lowerAtomicState_1);
            uint8_t lowerChargeState_2 = T_ConfigNumber::getIonizationState(
                lowerAtomicState_2);

            Idx upperAtomicState_1 = getUpperStateConfigNumber<Idx, T_Value>(tuple_1);
            Idx upperAtomicState_2 = getUpperStateConfigNumber<Idx, T_Value>(tuple_2);
            uint8_t upperChargeState_1 = T_ConfigNumber::getIonizationState(
                upperAtomicState_1);
            uint8_t upperChargeState_2 = T_ConfigNumber::getIonizationState(
                upperAtomicState_2);

            if constexpr(orderByLowerState)
            {
                if (lowerChargeState_1 != lowerChargeState_2)
                    return (lowerChargeState_1 < lowerChargeState_2);
                // lowerChargeState_1 == lowerChargeState_2
                if (lowerAtomicState_1 != lowerAtomicState_2)
                    return (lowerAtomicState_1 < lowerAtomicState_2);
                // lowerAtomicState_1 == lowerAtomicState_2
                if (upperChargeState_1 != upperChargeState_2)
                    return (upperChargeState_1 < lowerChargeState_2);
                // upperChargeState_1 == upperChargeState_2
                if (upperAtomicState_1 != upperAtomicState_2)
                    return (upperAtomicState_1 < upperAtomicState_2);

                // upperAtomicState_1 == upperAtomicState_2: --> all equal
                throw std::runtime_error("transitions with lower and upper state being equal are not allowed in the input data set!, Z: "
                    + std::to_string(T_ConfigNumber::atomicNumber));
            }
            else
            {
                if (upperChargeState_1 != upperChargeState_2)
                    return (upperChargeState_1 < lowerChargeState_2);
                // upperChargeState_1 == upperChargeState_2
                if (upperAtomicState_1 != upperAtomicState_2)
                    return (upperAtomicState_1 < upperAtomicState_2);
                // upperAtomicState_1 == upperAtomicState_2
                if (lowerChargeState_1 != lowerChargeState_2)
                    return (lowerChargeState_1 < lowerChargeState_2);
                // lowerChargeState_1 == lowerChargeState_2
                if (lowerAtomicState_1 != lowerAtomicState_2)
                    return (lowerAtomicState_1 < lowerAtomicState_2);

                // lowerAtomicState_1 == lowerAtomicState_2: --> all equal
                throw std::runtime_error("transitions with lower and upper state being equal are not"
                "allowed in the input data set!, Z: "
                    + std::to_string(T_ConfigNumber::atomicNumber));
            }
        }
    };
} // namespace picongpu::particles::atomicPhysics2::atomicData
