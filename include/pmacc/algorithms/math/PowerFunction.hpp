/* Copyright 2022 Brian Marre, Rene Widera, Richard Pausch
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#ifndef unitTest
#    define unitTest true
#endif

#include "pmacc/types.hpp"
#if unitTest == true
#    include "pmacc/static_assert.hpp"
#endif

#include <cstdint>

namespace pmacc::math
{
    /** power function for integer exponents, constexpr
     *
     * @tparam T_Type return and accumulation data type
     * @tparam T_Exp exponent data type, default uint32_t
     *
     * @param x base
     * @param exp exponent
     * @param result base multiplication
     */
    template<typename T_Type, typename T_Exp = uint32_t>
    HDINLINE constexpr T_Type pow(T_Type const x, T_Exp const exp)
    {
        T_Type result = static_cast<T_Type>(1u);
        for(T_Exp e = static_cast<T_Exp>(0u); e < exp; e++)
            result = result * x; // for whatever reason "*=" does not work, do not ask me ...
        return result;
    }

    namespace test
    {
#if unitTest == true
        PMACC_CASSERT_MSG(FAIL_unitTest_2_power_0, pow<uint32_t>(2u, 0u) == static_cast<uint32_t>(1u));
        PMACC_CASSERT_MSG(FAIL_unitTest_2_power_1, pow<uint8_t, uint8_t>(2u, 1u) == static_cast<uint8_t>(1u));
        PMACC_CASSERT_MSG(FAIL_unitTest_4_power_4, pow<uint32_t, uint8_t>(4u, 4u) == static_cast<uint32_t>(256u));
        PMACC_CASSERT_MSG(FAIL_unitTest_2_power_2, pow<double, uint8_t>(2., 2u) == 4.);
#endif
    } // namespace test

} // namespace pmacc::math
