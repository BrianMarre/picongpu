/* Copyright 2023 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software you can redistribute it andor modify
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

/** @file implements shorthand conversion function for enum to uint8_t
 *
 *  @attention do not use for enums with value ranges larger than uint8_t
 */

#pragma once

#include <cstdint>

namespace picongpu::particles::atomicPhysics2
{
    template<typename T_Enum>
    constexpr uint8_t u8(T_Enum const enumInstance)
    {
        return static_cast<uint8_t>(enumInstance);
    }
} // namespace picongpu::particles::atomicPhysics2
