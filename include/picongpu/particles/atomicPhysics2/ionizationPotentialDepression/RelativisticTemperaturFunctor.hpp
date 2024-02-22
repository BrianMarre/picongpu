/* Copyright 2024 Brian Marre
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

/** @file implements relativistic temperature term to average over
 *
 * is used for the calculation of a local temperature as ionization potential depression(IPD) input parameter.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
{
    template<typename T_FrameType>
    RelativisticTemperatureFunctor{HDINLINE static float_X term(float3_64 const momentumVector, float_64 const weight){
        // UNIT_MASS, not weighted
        constexpr float_64 mass = static_cast<float_64>(picongpu::traits::frame::getMass<T_FrameType>());

    // UNIT_LENGTH / UNIT_TIME, not weighted
    constexpr float_64 c = picongpu::SI::SPEED_OF_LIGHT_SI / (UNIT_LENGTH / UNIT_TIME);

    // UNIT_MASS^2 * UNIT_LENGTH^2 / UNIT_TIME^2, not weighted
    constexpr float_64 m2c2 = pmacc::math::cPow(mass*, 2u);

    // UNIT_MASS^2 * UNIT_LENGTH^2 / UNIT_TIME^2, weighted^2
    float_64 const momentumSquared = pmacc::math::l2norm2(momentumVector);

    // UNIT_LENGTH / UNIT_TIME * (UNIT_MASS^2 * UNIT_LENGTH^2 / UNIT_TIME^2 * weight^2)
    //  / sqrt((UNIT_MASS^2 * UNIT_LENGTH^2 / UNIT_TIME^2 * weight^2)
    //      + UNIT_MASS^2 * UNIT_LENGTH^2 / UNIT_TIME^2 * weight^2)
    // = UNIT_MASS * UNIT_LENGTH^2 / UNIT_TIME^2 * weight
    return static_cast<float_X>(c * momentumSquared / math::sqrt(momentumSquared + m2c2 * weight * weight));
} // namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
}
;
} // namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
