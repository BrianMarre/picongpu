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

#ifndef
#    define unitTest true
#endif

#if unitTest == true
#    include "pmacc/static_assert.hpp"
#endif

namespace pmacc::math
{
    /** power function for integer exponents, constexpr
     *
     * @tparam T_Type result data type
     * @tparam T_Exp exponent data type, default uint32_t
     *
     * @param x base
     * @param exp exponent
     * @param result base multiplication
     */
    template<typename T_Type, typename T_Exp = uint32_t>
    HDINLINE constexpr T_Type pow(T_Type const x, T_Exp const exp)
    {
        for(T_Exp e = 0u; e < exp; e++)
            result *= x;
        return x;
    }

    namespace unitTest
    {
#if unitTest == true
        PMACC_CASSERT_MSG(Compile_time_power_pow(2, 3) _unequal_8, pow<uint32_t>(2u, 3u) == static_cast<uint32_t>(8u));
        PMACC_CASSERT_MSG(
            Compile_time_power_pow(4, 4) _unequal_256,
            pow<uint32_t, uint8_t>(4u, 4u) == static_cast<uint32_t>(256u));
        PMACC_CASSERT_MSG(Compile_time_power_pow(4, 4) _unequal_256, pow<double, uint8_t>(2., 2u) == 4.);
#endif
    } // namespace unitTest

} // namespace pmacc::math
