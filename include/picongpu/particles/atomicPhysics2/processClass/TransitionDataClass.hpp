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

//! @file implements enum of the different kinds of transition data entries

#pragma once

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::processClass
{
    enum struct TransitionDataClass : uint8_t
    {
        boundBound_Up = 0u,
        boundBound_Down = 1u,
        boundFree_Up = 2u,
        boundFree_Down = 3u,
        autonomous_Up = 4u,
        autonomous_Down = 5u
    };
} // namespace picongpu::particles::atomicPhysics2::processClass
