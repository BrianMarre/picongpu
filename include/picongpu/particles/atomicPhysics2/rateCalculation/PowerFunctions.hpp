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

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::rateCalculation
{
    // complete compile time
    /** compile time construct for uint number exponentiation of uint numbers
     *
     * @attention beware of overflows!, uint8_t is small, check limits!
     *  uint8_t is sufficient for atomicPhysics, since inputs are at most 2 * 10^2 = 200 <= 255
     */
    template<uint8_t T_exponent, uint8_t T_base>
    struct CompileTimePower
    {
        constexpr uint8_t value = T_base * CompileTimePower<(T_exponent - 1u), T_base>::value;
    };

    //! partial specialization for early termination
    template<uint8_t T_base>
    struct CompileTimePower<1u, T_base>
    {
        constexpr uint8_t value = T_base;
    };

    //! partial specialization for termination
    template<uint8_t T_base>
    struct CompileTimePower<0u, T_base>
    {
        constexpr uint8_t value = 1u;
    };

    // base not known on compile time
    //! compile time uint power functor
    template<typename T_Type, uint8_t T_exponent>
    struct Power
    {
        HDINLINE static T_Type operator()(T_Type const base)
        {
            return base * Power<T_Type, static_cast<uint8_t>(T_exponent - 1u)>(base);
        }
    };

    //! partial specialization for early termination
    template<typename T_Type>
    struct Power<T_Type, 1u>
    {
        HDINLINE static T_Type operator()(T_Type const base)
        {
            return base;
        }
    };

    //! partial specialization for termination
    template<typename T_Type>
    struct Power<T_Type, 0u>
    {
        HDINLINE static T_Type operator()(T_Type const)
        {
            return 1u;
        }
    };

} // namespace picongpu::particles::atomicPhysics2::rateCalculation
