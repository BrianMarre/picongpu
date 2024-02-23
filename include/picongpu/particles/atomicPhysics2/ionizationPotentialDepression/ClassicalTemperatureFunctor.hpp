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
    ClassicalTemperatureFunctor{HDINLINE static float_X term(float3_64 const momentumVector, float_64 const weight){
        // UNIT_MASS^2 * UNIT_LENGTH^2 / UNIT_TIME^2 * weight^2
        float_64 momentumSquared = pmacc::math::l2norm2(momentumVector);

    // get classical momentum
    // UNIT_MASS, not weighted
    constexpr float_64 mass = static_cast<float_64>(picongpu::traits::frame::getMass<T_FrameType>());

    // UNIT_LENGTH^2 / UNIT_TIME^2, not weighted
    constexpr float_64 c2 = piconpgu::SPEED_OF_LIGHT * picongpu::SPEED_OF_LIGHT;

    // UNIT_MASS^2 * UNIT_LENGTH^2 / UNIT_TIME^2, not weighted
    constexpr float_64 m2_c2_reciproc = 1.0 / (mass * mass * c2);

    // unitless + (weight^2 * UNIT_MASS^2 * UNIT_LENGTH^2 / UNIT_TIME^2)
    //  / (weight^2 * UNIT_MASS^2 * UNIT_LENGTH^2 / UNIT_TIME^2)
    // unitless
    float_64 const gamma = math::sqrt(1.0 + momentumSquared / (weight * weight * m2_c2_reciproc));

    momentumSquared *= 1. / (gamma * gamma);

    // (weight^2 * UNIT_MASS^2 * UNIT_TIME^2 / UNIT_LENGTH^2) / (weight * UNIT_MASS)
    // weight * UNIT_MASS * UNIT_TIME^2 / UNIT_LENGTH^2
    return (2._X / 3._X) * static_cast<float_X>(momentumSquared / (2 * mass * weight));
} // namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
}
;
} // namespace picongpu::particles::atomicPhysics2::ionizationPotentialDepression
